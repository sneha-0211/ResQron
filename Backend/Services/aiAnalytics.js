import Drone from "../models/Drone.js";
import Mission from "../models/Mission.js";
import Disaster from "../models/Disaster.js";

export async function runAnalytics() {
  const drones = await Drone.find().exec();
  const missions = await Mission.find().exec();
  const disasters = await Disaster.find().exec();

  return {
    swarm: analyzeSwarmCoordination(drones),
    battery: analyzeBatteryStatus(drones),
    risk: assessOverallRisk(disasters),
    missions: optimizeMissionPlans(drones, disasters),
    maintenance: generateMaintenancePredictions(drones),
    recovery: generateRecoveryPlans(drones),
    weather: analyzeWeatherConditions(),
    terrain: optimizeTerrainNavigation(),
    energy: analyzeEnergyEfficiency(drones),
    intelligence: analyzeSwarmIntelligence(drones),
    prioritization: prioritizeMissions(drones, disasters),
    emergency: generateEmergencyProtocols(drones)
  };
}

// ---------------- Core analytics ----------------

function analyzeBatteryStatus(drones) {
  const low = drones.filter(d => d.battery < 30);
  const critical = drones.filter(d => d.battery < 20);
  return {
    low_count: low.length,
    critical_count: critical.length,
    recommendations: critical.length > 0 ? 
      ["Immediate RTL required for critical drones"] : ["Battery levels nominal"]
  };
}

function assessOverallRisk(disasters) {
  const highRisk = disasters.filter(d => d.severity === "high" || d.severity === "critical");
  return {
    total: disasters.length,
    high_risk: highRisk.length,
    level: highRisk.length > 0 ? "HIGH" : "MODERATE"
  };
}

function analyzeSwarmCoordination(drones) {
  const active = drones.filter(d => d.mode !== "IDLE").length;
  return {
    total: drones.length,
    active,
    coordination_score: active / (drones.length || 1),
    recommendations: active < drones.length / 2 ? 
      ["Increase swarm engagement for better coverage"] : []
  };
}

function generateMaintenancePredictions(drones) {
  return drones.map(d => {
    const score = d.battery / 100; 
    if (score < 0.5) {
      return {
        drone: d.callsign,
        issue: "Component degradation",
        severity: score < 0.3 ? "critical" : "warning",
        action: "Schedule maintenance soon"
      };
    }
    return null;
  }).filter(Boolean);
}

function generateRecoveryPlans(drones) {
  return drones.filter(d => d.battery < 20).map(d => ({
    drone: d.callsign,
    issue: "Low battery",
    plan: "Return to base & recharge",
    recovery_time: "15-30 min"
  }));
}

function optimizeMissionPlans(drones, disasters) {
  return disasters.map(dis => ({
    disaster: dis.type,
    assigned: drones.filter(d => dis.assignedDrones.includes(d.callsign)).length,
    recommendation: dis.priority === 1 ? "Deploy max drones ASAP" : "Monitor & reassign"
  }));
}

// Simulated AI parts
function analyzeWeatherConditions() {
  return {
    wind_speed: Math.random() * 20,
    visibility: Math.random() * 10 + 5,
    status: "ok"
  };
}

function optimizeTerrainNavigation() {
  return {
    complexity: "moderate",
    obstacles: Math.floor(Math.random() * 5),
    recommendation: "Adjust altitude if >3 obstacles detected"
  };
}

function analyzeEnergyEfficiency(drones) {
  const avg = drones.reduce((sum, d) => sum + (d.energy_efficiency || 1), 0) / (drones.length || 1);
  return {
    avg_efficiency: avg,
    waste: (1 - avg) * 100
  };
}

function analyzeSwarmIntelligence(drones) {
  return {
    decision_score: drones.filter(d => d.mode === "AUTO").length / (drones.length || 1),
    comm_quality: 0.9,
    coordination: 0.8
  };
}

function prioritizeMissions(drones, disasters) {
  return disasters.map(d => ({
    id: d._id,
    type: d.type,
    priority: d.priority || 5,
    recommendation: d.priority <= 2 ? "CRITICAL - deploy all drones" : "Normal response"
  }));
}

function generateEmergencyProtocols(drones) {
  const emergencies = drones.filter(d => d.battery < 15);
  return {
    active: emergencies.map(d => d.callsign),
    emergency_level: emergencies.length > 0 ? "critical" : "normal",
    recommendations: emergencies.length > 0 ? ["Activate RTL for low-battery drones"] : []
  };
}
