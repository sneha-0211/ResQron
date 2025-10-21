import express from "express";
import http from "http";
import { Server } from "socket.io";
import cors from "cors";
import bodyParser from "body-parser";
import mqtt from "mqtt";
import mongoose from "mongoose";
import dotenv from "dotenv";
import fetch from "node-fetch";
import Mission from "./models/Mission.js";
import Drone from "./models/Drone.js";
import EventModel from "./models/Event.js";
import NoFlyZone from "./models/NolyZone.js";
import Disaster from "./models/Disaster.js";

import { runAnalytics } from "./Services/aiAnalytics.js";

dotenv.config();

// Config
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/sih";
const MQTT_URL = process.env.MQTT_URL || "mqtt://localhost:1883";
const BATTERY_FAILSAFE = parseFloat(process.env.BATTERY_FAILSAFE || "20");
const MIN_BATTERY_ASSIGN = parseFloat(process.env.MIN_BATTERY_ASSIGN || "35");
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || null;
const PLANNER_URL = process.env.PLANNER_URL || "http://localhost:8000";

// MongoDB
await mongoose.connect(MONGO_URI).catch((err) => {
  console.error("Mongo connection error", err);
  process.exit(1);
});
console.log("MongoDB connected");

// Express + Socket.io
const app = express();
app.use(cors());
app.use(bodyParser.json());
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

// MQTT client
const mqttClient = mqtt.connect(MQTT_URL);
const pendingAcks = new Map();
const lastCommandAt = new Map();
const MIN_COMMAND_INTERVAL_MS = 200;

mqttClient.on("connect", () => {
  console.log("MQTT connected to", MQTT_URL);
  mqttClient.subscribe("drone/+/telemetry");
  mqttClient.subscribe("drone/+/event");
  mqttClient.subscribe("drone/+/ack");
  
  // Subscribe to AI perception systems
  mqttClient.subscribe("perception/+/detections");
  mqttClient.subscribe("depth/+/estimates");
  mqttClient.subscribe("obstacles/+/update");
  
  console.log("Subscribed to AI perception topics");
});

mqttClient.on("message", async (topic, message) => {
  try {
    const payload = JSON.parse(message.toString());

    // ---------------- Telemetry ----------------
    if (topic.match(/^drone\/[^\/]+\/telemetry$/)) {
      const callsign = topic.split("/")[1];
      const update = {
        callsign,
        battery: payload.battery ?? 100,
        mode: payload.mode ?? "IDLE",
        lastSeen: new Date(),
        location: { lat: payload.lat ?? 0, lng: payload.lng ?? 0, alt: payload.alt ?? 0 },
      };

      const drone = await Drone.findOneAndUpdate(
        { callsign },
        {
          $set: update,
          $push: { path: { $each: [[update.location.lat, update.location.lng]], $slice: -200 } },
        },
        { upsert: true, new: true }
      );

      // Failsafe RTL if battery low
      if (drone.battery <= BATTERY_FAILSAFE && drone.mode !== "RTL") {
        console.log(`Battery low for ${callsign} (${drone.battery}%) → RTL`);
        mqttClient.publish(`drone/${callsign}/command`, JSON.stringify({ cmd: "rtl" }));
        await Drone.updateOne({ callsign }, { $set: { mode: "RTL" } });

        const active = await Mission.findOne({ assignedTo: callsign, status: "active" });
        if (active) {
          active.assignedTo = null;
          active.status = "queued";
          await active.save();
          await tryAssignQueuedMissions();
          io.emit("mission-updated", active);
        }
      }

      io.emit("drone-update", drone);
    }

    // ---------------- Events ----------------
    if (topic.match(/^drone\/[^\/]+\/event$/)) {
      const callsign = topic.split("/")[1];
      const ev = new EventModel({ type: payload.event || "unknown", payload, source: callsign });
      await ev.save();
      io.emit("drone-event", { callsign, event: payload });

      // Human detected → auto create rescue mission
      if (payload.event === "human_detected" && payload.location) {
        const m = new Mission({
          name: "Rescue - human detected",
          waypoints: [[payload.location.lat, payload.location.lng]],
          supplies: ["first_aid", "water", "blanket"],
          priority: 1,
          metadata: { source: "ai", confidence: payload.confidence, image: payload.image },
        });
        await m.save();
        io.emit("mission-created", m);
        await tryAssignQueuedMissions();
      }
    }

    // ---------------- ACK ----------------
    if (topic.match(/^drone\/[^\/]+\/ack$/)) {
      const cmdId = payload?.cmdId;
      if (cmdId && pendingAcks.has(cmdId)) {
        const entry = pendingAcks.get(cmdId);
        clearTimeout(entry.timeout);
        pendingAcks.delete(cmdId);
        entry.resolve({ ok: true, status: payload.status || "ACK" });
      }
    }

    // ---------------- AI Perception Systems ----------------
    if (topic.match(/^perception\/[^\/]+\/detections$/)) {
      const cameraId = topic.split("/")[1];
      await handlePerceptionDetections(cameraId, payload);
    }

    if (topic.match(/^depth\/[^\/]+\/estimates$/)) {
      const cameraId = topic.split("/")[1];
      await handleDepthEstimates(cameraId, payload);
    }

    if (topic.match(/^obstacles\/[^\/]+\/update$/)) {
      const sourceId = topic.split("/")[1];
      await handleObstacleUpdate(sourceId, payload);
    }
  } catch (e) {
    console.error("MQTT msg parse error", e);
  }
});

// ---------------- AI Perception Handlers ----------------
async function handlePerceptionDetections(cameraId, payload) {
  try {
    console.log(`Received detections from camera ${cameraId}:`, payload.detections?.length || 0, "objects");
    
    // Store detection data for dashboard
    const detectionData = {
      cameraId,
      timestamp: payload.timestamp || Date.now(),
      detections: payload.detections || [],
      processingMs: payload.processing_ms || 0,
      model: payload.model || "unknown"
    };

    // Emit to dashboard
    io.emit("perception-detections", detectionData);

    // Check for critical detections that need immediate response
    const criticalDetections = payload.detections?.filter(d => 
      d.class === "person" && d.conf > 0.7
    ) || [];

    if (criticalDetections.length > 0) {
      console.log(`Critical detection: ${criticalDetections.length} person(s) detected by camera ${cameraId}`);
      
      // Create emergency rescue missions
      for (const detection of criticalDetections) {
        const mission = new Mission({
          name: `Emergency Rescue - Person detected by ${cameraId}`,
          waypoints: [[detection.lat || 34.0, detection.lng || -118.0]], // Simplified coordinates
          supplies: ["first_aid", "water", "blanket", "emergency_kit"],
          priority: 1,
          metadata: { 
            source: "ai_perception", 
            cameraId,
            confidence: detection.conf,
            detectionId: detection.id || "unknown",
            bbox: [detection.xmin, detection.ymin, detection.xmax, detection.ymax]
          },
        });
        await mission.save();
        io.emit("mission-created", mission);
        await tryAssignQueuedMissions();
      }
    }

    // Forward to planner for obstacle updates
    if (PLANNER_URL) {
      try {
        const plannerPayload = {
          camera_id: cameraId,
          detections: payload.detections,
          timestamp: payload.timestamp
        };
        
        // Publish to planner for obstacle mapping
        mqttClient.publish(`perception/${cameraId}/detections`, JSON.stringify(plannerPayload));
      } catch (e) {
        console.error("Failed to forward detections to planner:", e);
      }
    }

  } catch (e) {
    console.error("Error handling perception detections:", e);
  }
}

async function handleDepthEstimates(cameraId, payload) {
  try {
    console.log(`Received depth estimates from camera ${cameraId}:`, payload.depth_estimates?.length || 0, "objects");
    
    // Store depth data for dashboard
    const depthData = {
      cameraId,
      timestamp: payload.timestamp || Date.now(),
      depthEstimates: payload.depth_estimates || [],
      depthMapPath: payload.depth_map_path,
      processingMs: payload.processing_ms || 0,
      detectionCount: payload.detection_count || 0
    };

    // Emit to dashboard
    io.emit("depth-estimates", depthData);

    // Analyze depth data for obstacle proximity warnings
    const closeObstacles = payload.depth_estimates?.filter(d => 
      d.average_depth < 30 && d.depth_confidence === "high"
    ) || [];

    if (closeObstacles.length > 0) {
      console.log(`Close obstacles detected by camera ${cameraId}:`, closeObstacles.length);
      
      // Emit proximity warning
      io.emit("proximity-warning", {
        cameraId,
        closeObstacles: closeObstacles.length,
        timestamp: Date.now()
      });
    }

  } catch (e) {
    console.error("Error handling depth estimates:", e);
  }
}

async function handleObstacleUpdate(sourceId, payload) {
  try {
    console.log(`Received obstacle update from ${sourceId}:`, payload.obstacles?.length || 0, "obstacles");
    
    // Store obstacle data
    const obstacleData = {
      sourceId,
      timestamp: payload.timestamp || Date.now(),
      obstacles: payload.obstacles || [],
      obstacleCount: payload.obstacle_count || 0
    };

    // Emit to dashboard
    io.emit("obstacle-update", obstacleData);

    // Check if any active missions need replanning due to new obstacles
    const activeMissions = await Mission.find({ status: "active" });
    for (const mission of activeMissions) {
      // Simple check - in a real system, you'd do more sophisticated collision detection
      const needsReplanning = payload.obstacles?.some(obstacle => {
        // Check if obstacle is near mission path (simplified)
        return Math.abs(obstacle.lat - mission.waypoints[0][0]) < 0.001 && 
               Math.abs(obstacle.lon - mission.waypoints[0][1]) < 0.001;
      });

      if (needsReplanning) {
        console.log(`Mission ${mission._id} needs replanning due to new obstacles`);
        io.emit("mission-replan-needed", { missionId: mission._id, reason: "new_obstacles" });
      }
    }

  } catch (e) {
    console.error("Error handling obstacle update:", e);
  }
}

// ---------------- Mission Assignment Logic ----------------
async function tryAssignQueuedMissions() {
  const m = await Mission.findOne({ status: "queued" }).sort({ priority: 1, createdAt: 1 }).exec();
  if (!m) return;
  const drone = await chooseBestAvailableDrone();
  if (!drone) return;

  m.assignedTo = drone.callsign;
  m.status = "active";
  await m.save();

  mqttClient.publish(`mission/${m._id}/assign`, JSON.stringify(m));
  io.emit("mission-updated", m);
  console.log("Assigned mission", m._id, "to", drone.callsign);

  setImmediate(tryAssignQueuedMissions);
}

async function chooseBestAvailableDrone() {
  const drones = await Drone.find({ battery: { $gte: MIN_BATTERY_ASSIGN }, mode: { $ne: "RTL" } }).exec();
  const busy = await Mission.find({ status: "active" }).distinct("assignedTo").exec();
  const candidates = drones.filter((d) => !busy.includes(d.callsign));
  if (!candidates.length) return null;
  candidates.sort((a, b) => b.battery - a.battery);
  return candidates[0];
}

// ---------------- REST APIs ----------------
app.get("/api/drones", async (req, res) => res.json(await Drone.find().exec()));
app.get("/api/missions", async (req, res) => res.json(await Mission.find().sort({ createdAt: -1 }).limit(200).exec()));

// POST endpoint for creating missions (used by demo feeds)
app.post("/missions", async (req, res) => {
  try {
    const missionData = req.body;
    
    // Create new mission
    const mission = new Mission({
      name: missionData.name || 'Unnamed Mission',
      waypoints: missionData.waypoints || [],
      supplies: missionData.supplies || [],
      priority: missionData.priority || 5,
      assignedTo: missionData.assignedTo || null,
      status: missionData.status || 'queued',
      metadata: missionData.metadata || {}
    });
    
    const savedMission = await mission.save();
    
    // Emit to dashboard via Socket.io
    io.emit('mission_created', savedMission);
    
    // Publish to MQTT
    mqttClient.publish('missions/new', JSON.stringify(savedMission));
    
    console.log(`New mission created: ${savedMission._id} - ${savedMission.name}`);
    
    res.status(201).json(savedMission);
  } catch (error) {
    console.error('Error creating mission:', error);
    res.status(500).json({ error: 'Failed to create mission' });
  }
});

app.get("/api/disasters", async (req, res) => res.json(await Disaster.find().sort({ detectedAt: -1 }).limit(100).exec()));

// POST endpoint for creating disasters (used by demo feeds)
app.post("/disasters", async (req, res) => {
  try {
    const disasterData = req.body;
    
    // Create new disaster
    const disaster = new Disaster({
      type: disasterData.type || 'unknown',
      severity: disasterData.severity || 'moderate',
      confidence: disasterData.confidence || 0.5,
      coordinates: disasterData.coordinates || { lat: 0, lng: 0 },
      description: disasterData.description || '',
      recommendedActions: disasterData.recommendedActions || [],
      status: disasterData.status || 'detected',
      assignedDrones: disasterData.assignedDrones || [],
      imageUrl: disasterData.imageUrl || '',
      detectedAt: disasterData.detectedAt ? new Date(disasterData.detectedAt) : new Date(),
      metadata: disasterData.metadata || {}
    });
    
    const savedDisaster = await disaster.save();
    
    // Emit to dashboard via Socket.io
    io.emit('disaster_detected', savedDisaster);
    
    // Publish to MQTT
    mqttClient.publish('disasters/new', JSON.stringify(savedDisaster));
    
    console.log(`New disaster created: ${savedDisaster._id} - ${savedDisaster.type}`);
    
    res.status(201).json(savedDisaster);
  } catch (error) {
    console.error('Error creating disaster:', error);
    res.status(500).json({ error: 'Failed to create disaster' });
  }
});

app.get("/api/events", async (req, res) => res.json(await EventModel.find().sort({ createdAt: -1 }).limit(200).exec()));

// ---------------- AI Integration APIs ----------------
app.get("/api/perception/status", async (req, res) => {
  try {
    // Get recent detection data
    const recentDetections = await EventModel.find({ 
      type: "perception_detection" 
    }).sort({ createdAt: -1 }).limit(50).exec();
    
    res.json({
      status: "operational",
      recentDetections: recentDetections.length,
      lastUpdate: recentDetections[0]?.createdAt || null,
      mqttConnected: mqttClient.connected
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/api/perception/request", async (req, res) => {
  try {
    const { cameraId, imageData, requestType = "detection" } = req.body;
    
    if (!cameraId || !imageData) {
      return res.status(400).json({ error: "cameraId and imageData are required" });
    }

    // Forward request to AI services via MQTT
    const requestPayload = {
      camera_id: cameraId,
      image_data: imageData,
      request_type: requestType,
      timestamp: Date.now()
    };

    mqttClient.publish(`ai/${cameraId}/request`, JSON.stringify(requestPayload));
    
    res.json({ 
      message: "Request sent to AI services",
      cameraId,
      requestType 
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/obstacles", async (req, res) => {
  try {
    // Get obstacle data from planner service
    if (PLANNER_URL) {
      const response = await fetch(`${PLANNER_URL}/obstacles`);
      const obstacleData = await response.json();
      res.json(obstacleData);
    } else {
      res.json({ 
        obstacle_count: 0, 
        obstacles: [],
        message: "Planner service not configured" 
      });
    }
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/geofences", async (req, res) => {
  try {
    // Get geofence data from planner service
    if (PLANNER_URL) {
      const response = await fetch(`${PLANNER_URL}/geofences`);
      const geofenceData = await response.json();
      res.json(geofenceData);
    } else {
      res.json({ 
        geofence_count: 0, 
        geofences: [],
        message: "Planner service not configured" 
      });
    }
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/api/plan/mission", async (req, res) => {
  try {
    const { droneId, start, goal, useDynamicObstacles = true } = req.body;
    
    if (!droneId || !start || !goal) {
      return res.status(400).json({ error: "droneId, start, and goal are required" });
    }

    // Forward to planner service
    if (PLANNER_URL) {
      const plannerResponse = await fetch(`${PLANNER_URL}/plan_direct`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start,
          goal,
          use_dynamic_obstacles: useDynamicObstacles
        })
      });
      
      const pathData = await plannerResponse.json();
      res.json(pathData);
    } else {
      res.status(503).json({ error: "Planner service not available" });
    }
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/system/health", async (req, res) => {
  try {
    const healthData = {
      backend: {
        status: "healthy",
        mqttConnected: mqttClient.connected,
        websocketClients: io.engine.clientsCount,
        uptime: process.uptime()
      },
      ai: {
        perceptionService: AI_SERVICE_URL ? "configured" : "not_configured",
        plannerService: PLANNER_URL ? "configured" : "not_configured"
      },
      database: {
        status: "connected",
        droneCount: await Drone.countDocuments(),
        missionCount: await Mission.countDocuments(),
        eventCount: await EventModel.countDocuments()
      },
      timestamp: Date.now()
    };

    res.json(healthData);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Drone commands (Takeoff, Land, RTL, etc.)
app.post("/api/drone/:callsign/command", async (req, res) => {
  try {
    const { callsign } = req.params;
    const { cmd, meta } = req.body;
    const now = Date.now();
    const last = lastCommandAt.get(callsign) || 0;
    if (now - last < MIN_COMMAND_INTERVAL_MS) return res.status(429).json({ ok: false, msg: "rate_limited" });

    lastCommandAt.set(callsign, now);
    const cmdId = `${callsign}_${now}_${Math.floor(Math.random() * 1e6)}`;
    const payload = { cmdId, cmd, meta };

    const awaitAck = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        pendingAcks.delete(cmdId);
        reject(new Error("ack_timeout"));
      }, 3000);
      pendingAcks.set(cmdId, { resolve, reject, timeout });
    });

    mqttClient.publish(`drone/${callsign}/command`, JSON.stringify(payload));
    await awaitAck;
    res.json({ ok: true, cmdId });
  } catch (e) {
    res.status(500).json({ ok: false, err: e.toString() });
  }
});

// ---------------- WebSocket ----------------
io.on("connection", async (socket) => {
  console.log("Frontend connected", socket.id);

  // Send initial system state
  try {
    socket.emit("system-status", {
      drones: await Drone.find().exec(),
      missions: await Mission.find().sort({ createdAt: -1 }).limit(50).exec(),
      timestamp: Date.now()
    });
  } catch (error) {
    console.error("Error sending initial system state:", error);
    socket.emit("system-status", {
      drones: [],
      missions: [],
      timestamp: Date.now()
    });
  }

  socket.on("command", ({ droneId, cmd }) => {
    if (!droneId || !cmd) return;
    mqttClient.publish(`drone/${droneId}/command`, JSON.stringify({ cmd }));
    console.log("MQTT command", droneId, cmd);
  });

  socket.on("return-home", (droneId) => {
    mqttClient.publish(`drone/${droneId}/command`, JSON.stringify({ cmd: "rtl" }));
  });

  socket.on("enable-autonomous", (droneId) => {
    mqttClient.publish(`drone/${droneId}/command`, JSON.stringify({ cmd: "autonomous_on" }));
  });

  socket.on("disable-autonomous", (droneId) => {
    mqttClient.publish(`drone/${droneId}/command`, JSON.stringify({ cmd: "autonomous_off" }));
  });

  socket.on("assign-zone", ({ droneId, zoneId, priority }) => {
    mqttClient.publish(`drone/${droneId}/command`, JSON.stringify({ cmd: "assign_zone", zone: zoneId, priority }));
  });

  // AI-related WebSocket events
  socket.on("request-perception", async ({ cameraId, imageData }) => {
    try {
      const requestPayload = {
        camera_id: cameraId,
        image_data: imageData,
        request_type: "detection",
        timestamp: Date.now()
      };
      
      mqttClient.publish(`ai/${cameraId}/request`, JSON.stringify(requestPayload));
      socket.emit("perception-request-sent", { cameraId, timestamp: Date.now() });
    } catch (e) {
      socket.emit("error", { message: "Failed to send perception request", error: e.message });
    }
  });

  socket.on("request-depth", async ({ cameraId, imageData, detections }) => {
    try {
      const requestPayload = {
        camera_id: cameraId,
        image_data: imageData,
        detections: detections || [],
        request_type: "depth",
        timestamp: Date.now()
      };
      
      mqttClient.publish(`depth/${cameraId}/request`, JSON.stringify(requestPayload));
      socket.emit("depth-request-sent", { cameraId, timestamp: Date.now() });
    } catch (e) {
      socket.emit("error", { message: "Failed to send depth request", error: e.message });
    }
  });

  socket.on("plan-mission", async ({ droneId, start, goal, useDynamicObstacles = true }) => {
    try {
      if (!droneId || !start || !goal) {
        socket.emit("error", { message: "droneId, start, and goal are required" });
        return;
      }

      if (PLANNER_URL) {
        const response = await fetch(`${PLANNER_URL}/plan_direct`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            start,
            goal,
            use_dynamic_obstacles: useDynamicObstacles
          })
        });
        
        const pathData = await response.json();
        socket.emit("mission-planned", { droneId, path: pathData });
      } else {
        socket.emit("error", { message: "Planner service not available" });
      }
    } catch (e) {
      socket.emit("error", { message: "Failed to plan mission", error: e.message });
    }
  });

  socket.on("update-obstacles", async ({ obstacles }) => {
    try {
      if (PLANNER_URL) {
        const response = await fetch(`${PLANNER_URL}/update_obstacles`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(obstacles)
        });
        
        const result = await response.json();
        socket.emit("obstacles-updated", result);
      } else {
        socket.emit("error", { message: "Planner service not available" });
      }
    } catch (e) {
      socket.emit("error", { message: "Failed to update obstacles", error: e.message });
    }
  });

  socket.on("get-system-status", async () => {
    try {
      const healthData = {
        backend: {
          status: "healthy",
          mqttConnected: mqttClient.connected,
          websocketClients: io.engine.clientsCount,
          uptime: process.uptime()
        },
        ai: {
          perceptionService: AI_SERVICE_URL ? "configured" : "not_configured",
          plannerService: PLANNER_URL ? "configured" : "not_configured"
        },
        database: {
          status: "connected",
          droneCount: await Drone.countDocuments(),
          missionCount: await Mission.countDocuments(),
          eventCount: await EventModel.countDocuments()
        },
        timestamp: Date.now()
      };
      
      socket.emit("system-status", healthData);
    } catch (e) {
      socket.emit("error", { message: "Failed to get system status", error: e.message });
    }
  });

  socket.on("disconnect", () => console.log("Frontend disconnected", socket.id));
});

// ---------------- Analytics Loop ----------------
setInterval(async () => {
  try {
    const analytics = await runAnalytics();
    io.emit("system-health", analytics);
  } catch (e) {
    console.error("Analytics error", e);
  }
}, 10000); // every 10s

// ---------------- Start ----------------
server.listen(PORT, () => console.log(`Backend listening on ${PORT}`));
setInterval(tryAssignQueuedMissions, 5000);
