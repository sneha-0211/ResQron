import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from "framer-motion";
import { Bot, ShieldCheck, SlidersHorizontal, TriangleAlert, CheckCircle2, XCircle, CircleDashed, Home } from 'lucide-react';
import GoogleMapComponent from './GoogleMapComponent';

// Notification Component  
const Notification = ({ message, type }) => {
  return (
    <motion.div
      className={`notification-item ${type}`}
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 100, opacity: 0 }}
      transition={{ duration: 0.4, ease: "easeInOut" }}
      layout
    >
      {message}
    </motion.div>
  );
};

// Drone Icon  
const DroneIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="logo-icon" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 12L8 8M12 12L16 8M12 12L8 16M12 12L16 16" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 21C16.9706 21 21 16.9706 21 12C21 7.02944 16.9706 3 12 3C7.02944 3 3 7.02944 3 12C3 16.9706 7.02944 21 12 21Z" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 12L19 5" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 12L5 19" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 12L5 5" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 12L19 19" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
);


// Left Sidebar component updated with Tooltips
const LeftSidebar = ({ addNotification, onSwarmActivated, onRTLActivated }) => {
  const [isRiskAssessmentOn, setIsRiskAssessmentOn] = useState(true);
  const [drones, setDrones] = useState([
    {
      id: 1,
      name: "Alpha",
      status: "active",
      battery: 85,
      altitude: 120,
      speed: 45,
      mission: "Search & Rescue",
      lastUpdate: new Date().toLocaleTimeString()
    },
    {
      id: 2,
      name: "Gamma", 
      status: "active",
      battery: 92,
      altitude: 95,
      speed: 38,
      mission: "Area Surveillance",
      lastUpdate: new Date().toLocaleTimeString()
    },
    {
      id: 3,
      name: "Beta",
      status: "maintenance",
      battery: 15,
      altitude: 0,
      speed: 0,
      mission: "Standby",
      lastUpdate: new Date().toLocaleTimeString()
    }
  ]);

  const actions = [
    { label: "Swarm Coord", icon: <Bot size={28} />, isEmergency: false, message: "Swarm Coordination Activated!", action: "swarm" },
    { label: "Predictive Maint", icon: <ShieldCheck size={28} />, isEmergency: false, message: "Predictive Maintenance Analysis Started.", action: "maintenance" },
    { label: "Reroute Optimize", icon: <SlidersHorizontal size={28} />, isEmergency: false, message: "Drone Rerouting Optimized.", action: "reroute" },
    { label: "Emergency Override", icon: <TriangleAlert size={28} />, isEmergency: true, message: "Emergency Override Engaged!", action: "emergency" },
  ];

  const handleActionClick = (action) => {
    const type = action.isEmergency ? 'emergency' : 'success';
    
    // Handle swarm action specially - only show one notification
    if (action.action === 'swarm') {
      addNotification("Swarm Activated - 3 drones coordinating...", 'success');
      onSwarmActivated();
    } else {
      addNotification(action.message, type);
    }
  };

  const handleRTL = () => {
    addNotification("Returning to Lobby...", 'success');
    onRTLActivated();
  };

  // Update drone data periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setDrones(prevDrones => 
        prevDrones.map(drone => {
          if (drone.status === 'active') {
            return {
              ...drone,
              battery: Math.max(10, drone.battery - Math.random() * 2),
              lastUpdate: new Date().toLocaleTimeString()
            };
          }
          return drone;
        })
      );
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="sidebar">
      <div className="header">
        <DroneIcon />
        <h1>ResQron</h1>
      </div>
      <div className="control-panel">
        <div className="risk-assessment">
          <span>Risk Assessment</span>
          <label className="toggle-switch">
            <input type="checkbox" checked={isRiskAssessmentOn} onChange={() => setIsRiskAssessmentOn(!isRiskAssessmentOn)} />
            <span className="slider"></span>
          </label>
        </div>
        <div className="actions-grid">
          {actions.map((action) => (
            // **CHANGE**: Each button is now wrapped in a div for the tooltip
            <div key={action.label} className="tooltip-wrapper">
              <button 
                className={`circular-button ${action.isEmergency ? 'emergency' : ''}`}
                onClick={() => handleActionClick(action)}
              >
                {action.icon}
                {/* Text label removed from here */}
              </button>
              <span className="tooltip">{action.label}</span>
            </div>
          ))}
          
          {/* RTL Button in center */}
          <div className="tooltip-wrapper rtl-button-wrapper">
            <button 
              className="circular-button rtl-button"
              onClick={handleRTL}
            >
              <Home size={28} />
            </button>
            <span className="tooltip">RTL (Return to Lobby)</span>
          </div>
        </div>
      </div>

      {/* Drone Fleet Status Panel */}
      <div className="drone-fleet-panel">
        <h4 className="panel-header">Drone Fleet Status</h4>
        <div className="drone-status-list">
          {drones.map(drone => (
            <div key={drone.id} className="drone-status-item">
              <div className="drone-status-header">
                <span className="drone-name">{drone.name}</span>
                <span className={`drone-status ${drone.status}`}>
                  {drone.status.toUpperCase()}
                </span>
              </div>
              <div className="drone-status-details">
                <span>Battery: {drone.battery.toFixed(1)}%</span>
                <span>Alt: {drone.altitude}m</span>
              </div>
              <div className="drone-mission">
                <span>Mission: {drone.mission}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};


// Right Sidebar component  
const RightSidebar = () => {
  // Placeholder data
  const systemHealthData = [
    { name: "Drone Connectivity", percentage: 98, status: "status-green" },
    { name: "Sensor Calibration", percentage: 92, status: "status-green" },
    { name: "GPS Lock Accuracy", percentage: 95, status: "status-green" },
  ];

  const missionStatusData = [
    { name: "Pre-flight Check", status: "completed", icon: <CheckCircle2 className="status-green"/> },
    { name: "Take-off & Ascent", status: "completed", icon: <CheckCircle2 className="status-green"/> },
    { name: "En Route to Target", status: "in_progress", icon: <CircleDashed className="status-orange" /> },
    { name: "Area Scan", status: "pending", icon: <XCircle className="status-red"/> },
  ];

  return (
    <div className="sidebar">
      <div className="info-panel">
        <h2 className="panel-header">System Health & Prediction</h2>
        {systemHealthData.map(item => (
          <div key={item.name}>
            <div className="status-item">
              <span>{item.name}</span>
              <span className={item.status}>{item.percentage}%</span>
            </div>
            <div className="progress-bar-container">
              <div className="progress-bar" style={{ width: `${item.percentage}%` }}></div>
            </div>
          </div>
        ))}
      </div>

      <div className="info-panel">
        <h2 className="panel-header">Failure Prediction</h2>
        <div className="status-item">
          <span>Rotor Integrity</span>
          <span className="status-green">Optimal</span>
        </div>
        <div className="status-item">
          <span>Battery Optimization</span>
          <span className="status-orange">91% (Degrading)</span>
        </div>
         <div className="status-item">
          <span>Gimbal Motor Stress</span>
          <span className="status-green">Low</span>
        </div>
      </div>

      <div className="info-panel">
        <h2 className="panel-header">Mission Status</h2>
        <ul className="mission-list">
          {missionStatusData.map(item => (
            <li key={item.name}>
              {item.icon} {item.name}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};


// Main CommandCenter Component (No changes in logic)
function CommandCenter() {
  const [notifications, setNotifications] = useState([]);
  const [swarmTrigger, setSwarmTrigger] = useState(0);
  const [rtlTrigger, setRTLTrigger] = useState(0);

  const addNotification = useCallback((message, type) => {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, message, type }]);

    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 10000);
  }, []);

  const handleSwarmActivated = useCallback(() => {
    // Trigger swarm activation
    setSwarmTrigger(prev => prev + 1);
  }, []);

  const handleRTLActivated = useCallback(() => {
    // Trigger RTL activation
    setRTLTrigger(prev => prev + 1);
    addNotification("RTL activated - returning to lobby!", 'success');
  }, [addNotification]);

  return (
    <>
      <LeftSidebar 
        addNotification={addNotification} 
        onSwarmActivated={handleSwarmActivated}
        onRTLActivated={handleRTLActivated}
      />
      
      <div className="main-content">
        <div className="notification-container">
          <AnimatePresence>
            {notifications.map(n => (
              <Notification key={n.id} message={n.message} type={n.type} />
            ))}
          </AnimatePresence>
        </div>
        <GoogleMapComponent 
          onSwarmActivated={handleSwarmActivated}
          onRTLActivated={handleRTLActivated}
          swarmTrigger={swarmTrigger}
          rtlTrigger={rtlTrigger}
        />
      </div>

      <RightSidebar />
    </>
  );
}

export default CommandCenter;
