import React, { useState, useEffect, useRef, useMemo } from 'react';
import { GoogleMap, LoadScript, Marker, InfoWindow, Polyline, Autocomplete } from '@react-google-maps/api';

const GoogleMapComponent = ({ onSwarmActivated, onRTLActivated, swarmTrigger, rtlTrigger }) => {
  const mapRef = useRef(null);
  const [map, setMap] = useState(null);
  const [selectedDrone, setSelectedDrone] = useState(null);
  const [dronePositions, setDronePositions] = useState([]);
  const [flightPaths, setFlightPaths] = useState([]);
  const [userLocation, setUserLocation] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [locationError, setLocationError] = useState(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [useStaticMap, setUseStaticMap] = useState(false);
  const [autocomplete, setAutocomplete] = useState(null);
  const [searchMarker, setSearchMarker] = useState(null);
  const inputRef = useRef(null);
  const [searchValue, setSearchValue] = useState("");
  const [swarmDrone, setSwarmDrone] = useState(null);
  const [swarmRoute, setSwarmRoute] = useState(null);
  const [isRTLActive, setIsRTLActive] = useState(false);
  const [swarmDrones, setSwarmDrones] = useState([]);
  const [isSwarmActive, setIsSwarmActive] = useState(false);

  // Default center (Delhi, India) - fallback if geolocation fails
  const defaultCenter = {
    lat: 28.6139,
    lng: 77.2090
  };

  // Odisha coordinates (Bhubaneswar)
  const odishaLocation = {
    lat: 20.2961,
    lng: 85.8245
  };

  // Sample drone data with realistic positions
  const [drones, setDrones] = useState([
    {
      id: 1,
      name: "Alpha",
      position: { lat: 28.6139, lng: 77.2090 },
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
      position: { lat: 28.6145, lng: 77.2105 },
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
      position: { lat: 28.6125, lng: 77.2075 },
      status: "maintenance",
      battery: 15,
      altitude: 0,
      speed: 0,
      mission: "Standby",
      lastUpdate: new Date().toLocaleTimeString()
    }
  ]);

  // Map container styles
  const mapContainerStyle = {
    width: '100%',
    height: '100%',
    borderRadius: '12px'
  };

  // Simplified map options for faster loading
  const mapOptions = useMemo(() => ({
    styles: [
      { featureType: "all", elementType: "geometry", stylers: [{ color: "#242f3e" }] },
      { featureType: "all", elementType: "labels.text.fill", stylers: [{ color: "#746855" }] },
      { featureType: "water", elementType: "geometry", stylers: [{ color: "#17263c" }] },
      { featureType: "road", elementType: "geometry", stylers: [{ color: "#38414e" }] }
    ],
    disableDefaultUI: true,
    zoomControl: true,
    streetViewControl: false,
    mapTypeControl: false,
    fullscreenControl: false,
    gestureHandling: 'greedy',
    clickableIcons: false,
    keyboardShortcuts: false,
    disableDoubleClickZoom: false,
    scrollwheel: true,
    draggable: true
  }), []);

  // Custom drone marker icon (safe before Maps is loaded)
  const getDroneIcon = (status, battery) => {
    const color = status === 'active' ? (battery > 50 ? '#00ff7f' : '#ffaa00') : '#ff4d4d';
    const anchorPoint = (typeof window !== 'undefined' && window.google && window.google.maps)
      ? new window.google.maps.Point(12, 12)
      : undefined;
    return {
      path: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z',
      fillColor: color,
      fillOpacity: 1,
      strokeColor: '#ffffff',
      strokeWeight: 2,
      scale: 1.5,
      anchor: anchorPoint
    };
  };

  // Get user's current location with faster timeout
  useEffect(() => {
    // Set a maximum loading time of 8 seconds
    const loadingTimeout = setTimeout(() => {
      setIsLoading(false);
    }, 8000);

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          clearTimeout(loadingTimeout);
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
          setIsLoading(false);
        },
        (error) => {
          clearTimeout(loadingTimeout);
          console.warn('Geolocation error:', error);
          setLocationError('Unable to get your location. Using default location.');
          setIsLoading(false);
        },
        {
          enableHighAccuracy: false, // Faster, less accurate
          timeout: 3000, // Reduced to 3s
          maximumAge: 600000 // 10 minutes cache
        }
      );
    } else {
      clearTimeout(loadingTimeout);
      setLocationError('Geolocation is not supported by this browser.');
      setIsLoading(false);
    }

    return () => clearTimeout(loadingTimeout);
  }, []);

  // Update drone positions periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setDrones(prevDrones => 
        prevDrones.map(drone => {
          if (drone.status === 'active') {
            // Simulate drone movement
            const latOffset = (Math.random() - 0.5) * 0.001;
            const lngOffset = (Math.random() - 0.5) * 0.001;
            
            return {
              ...drone,
              position: {
                lat: drone.position.lat + latOffset,
                lng: drone.position.lng + lngOffset
              },
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

  // Handle map load
  const onMapLoad = (mapInstance) => {
    setMap(mapInstance);
    mapRef.current = mapInstance;
    setMapLoaded(true);
  };

  // Handle marker click
  const onMarkerClick = (drone) => {
    setSelectedDrone(drone);
  };

  // Close info window
  const onInfoWindowClose = () => {
    setSelectedDrone(null);
  };

  // Autocomplete handlers
  const onAutocompleteLoad = (acInstance) => {
    setAutocomplete(acInstance);
    // Optionally bias to current viewport/user location for better accuracy
    if (mapRef.current && acInstance.setBounds) {
      acInstance.setBounds(mapRef.current.getBounds());
    }
  };

  const onPlaceChanged = () => {
    if (!autocomplete) return;
    const place = autocomplete.getPlace();
    if (!place || !place.geometry || !place.geometry.location) return;
    const location = {
      lat: place.geometry.location.lat(),
      lng: place.geometry.location.lng()
    };
    if (mapRef.current) {
      mapRef.current.panTo(location);
      mapRef.current.setZoom(16);
    }
    setSearchMarker(location);
    // Set input value to exact formatted address/name
    const newValue = place.formatted_address || place.name || "";
    if (newValue) setSearchValue(newValue);
  };

  // Manual geocode for exact text search (addresses/coordinates) on Enter
  const geocodeAddress = (query) => {
    if (!query) return;
    if (!(window.google && window.google.maps)) return;
    const geocoder = new window.google.maps.Geocoder();
    geocoder.geocode({ address: query }, (results, status) => {
      if (status === 'OK' && results && results[0]) {
        const loc = results[0].geometry.location;
        const location = { lat: loc.lat(), lng: loc.lng() };
        if (mapRef.current) {
          mapRef.current.panTo(location);
          mapRef.current.setZoom(17);
        }
        setSearchMarker(location);
      }
    });
  };

  const onSearchKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const val = searchValue || (inputRef.current ? inputRef.current.value : "");
      geocodeAddress(val);
    }
  };

  // Swarm functionality - create 3 animated drones coordinating together
  const createSwarmRoute = () => {
    const currentLocation = userLocation || defaultCenter;
    setIsSwarmActive(true);
    
    // Create 3 swarm drones with different starting positions
    const newSwarmDrones = [
      {
        id: 'swarm-001',
        name: 'Swarm-001',
        position: { lat: currentLocation.lat + 0.001, lng: currentLocation.lng + 0.001 },
        status: 'active',
        battery: 100,
        altitude: 150,
        speed: 60,
        mission: 'Swarm Coordination',
        lastUpdate: new Date().toLocaleTimeString(),
        color: '#00ff7f'
      },
      {
        id: 'swarm-002',
        name: 'Swarm-002',
        position: { lat: currentLocation.lat - 0.001, lng: currentLocation.lng + 0.001 },
        status: 'active',
        battery: 95,
        altitude: 160,
        speed: 58,
        mission: 'Swarm Coordination',
        lastUpdate: new Date().toLocaleTimeString(),
        color: '#00bfff'
      },
      {
        id: 'swarm-003',
        name: 'Swarm-003',
        position: { lat: currentLocation.lat, lng: currentLocation.lng - 0.001 },
        status: 'active',
        battery: 98,
        altitude: 140,
        speed: 62,
        mission: 'Swarm Coordination',
        lastUpdate: new Date().toLocaleTimeString(),
        color: '#ff6b6b'
      }
    ];
    
    setSwarmDrones(newSwarmDrones);
    
    // Start coordination animation
    startSwarmCoordination(newSwarmDrones, currentLocation);
  };

  // Start swarm coordination animation
  const startSwarmCoordination = (drones, centerLocation) => {
    let animationStep = 0;
    const maxSteps = 20; // Number of coordination steps
    
    const coordinationInterval = setInterval(() => {
      setSwarmDrones(prevDrones => 
        prevDrones.map((drone, index) => {
          // Create circular coordination pattern
          const angle = (animationStep * 0.3) + (index * 2.09); // 2.09 ≈ 2π/3 for 120° spacing
          const radius = 0.002 + Math.sin(animationStep * 0.2) * 0.001; // Varying radius
          
          const newLat = centerLocation.lat + Math.cos(angle) * radius;
          const newLng = centerLocation.lng + Math.sin(angle) * radius;
          
          return {
            ...drone,
            position: { lat: newLat, lng: newLng },
            altitude: 150 + Math.sin(animationStep * 0.3 + index) * 20,
            battery: Math.max(70, drone.battery - 0.5),
            lastUpdate: new Date().toLocaleTimeString()
          };
        })
      );
      
      animationStep++;
      
      if (animationStep >= maxSteps) {
        clearInterval(coordinationInterval);
        // After coordination, start moving towards Odisha
        startSwarmMission();
      }
    }, 1000);
  };

  // Start swarm mission to Odisha
  const startSwarmMission = () => {
    const currentLocation = userLocation || defaultCenter;
    const routePath = [
      currentLocation,
      { lat: currentLocation.lat - 2, lng: currentLocation.lng + 1 },
      { lat: currentLocation.lat - 4, lng: currentLocation.lng + 2 },
      { lat: currentLocation.lat - 6, lng: currentLocation.lng + 3 },
      odishaLocation
    ];
    
    setSwarmRoute({
      coordinates: routePath,
      color: '#00ff7f'
    });
    
    // Animate swarm drones moving to Odisha
    animateSwarmToOdisha(routePath);
  };

  // Animate swarm drones moving to Odisha
  const animateSwarmToOdisha = (route) => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex < route.length) {
        setSwarmDrones(prevDrones => 
          prevDrones.map(drone => ({
            ...drone,
            position: route[currentIndex],
            battery: Math.max(20, drone.battery - 2),
            mission: 'En Route to Odisha',
            lastUpdate: new Date().toLocaleTimeString()
          }))
        );
        currentIndex++;
      } else {
        clearInterval(interval);
        // Drones reached Odisha
        setSwarmDrones(prevDrones => 
          prevDrones.map(drone => ({
            ...drone,
            mission: 'Mission Complete - Odisha',
            status: 'completed'
          }))
        );
        setIsSwarmActive(false);
      }
    }, 2000);
  };

  // RTL functionality - return swarm drones to lobby
  const activateRTL = () => {
    if (swarmDrones.length > 0) {
      setIsRTLActive(true);
      const lobbyLocation = userLocation || defaultCenter;
      
      // Create return route from current swarm position
      const currentPosition = swarmDrones[0].position;
      const returnRoute = [
        currentPosition,
        { lat: currentPosition.lat + 2, lng: currentPosition.lng - 1 },
        { lat: currentPosition.lat + 4, lng: currentPosition.lng - 2 },
        { lat: currentPosition.lat + 6, lng: currentPosition.lng - 3 },
        lobbyLocation
      ];
      
      setSwarmRoute({
        coordinates: returnRoute,
        color: '#ffaa00'
      });
      
      // Animate return journey
      animateRTLReturn(returnRoute);
    }
  };

  // Animate RTL return
  const animateRTLReturn = (route) => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex < route.length) {
        setSwarmDrones(prevDrones => 
          prevDrones.map(drone => ({
            ...drone,
            position: route[currentIndex],
            battery: Math.max(10, drone.battery - 1),
            mission: 'RTL - Returning to Lobby',
            lastUpdate: new Date().toLocaleTimeString()
          }))
        );
        currentIndex++;
      } else {
        clearInterval(interval);
        // Drones returned to lobby
        setSwarmDrones(prevDrones => 
          prevDrones.map(drone => ({
            ...drone,
            mission: 'RTL Complete - At Lobby',
            status: 'standby'
          }))
        );
        setIsRTLActive(false);
        setSwarmRoute(null);
      }
    }, 1500);
  };

  // Listen for swarm and RTL activation
  useEffect(() => {
    if (swarmTrigger) {
      createSwarmRoute();
    }
  }, [swarmTrigger]);

  useEffect(() => {
    if (rtlTrigger) {
      activateRTL();
    }
  }, [rtlTrigger]);

  // Determine map center
  const mapCenter = userLocation || defaultCenter;

  // Show loading state
  if (isLoading) {
    return (
      <div className="google-map-container">
        <div className="map-loading">
          <div className="loading-spinner"></div>
          <p>Loading map...</p>
          <button 
            className="fallback-button"
            onClick={() => setUseStaticMap(true)}
          >
            Use Static Map (Faster)
          </button>
        </div>
      </div>
    );
  }

  // Show static map fallback
  if (useStaticMap) {
    const mapCenter = userLocation || defaultCenter;
    const staticMapUrl = `https://maps.googleapis.com/maps/api/staticmap?center=${mapCenter.lat},${mapCenter.lng}&zoom=15&size=800x600&maptype=roadmap&style=feature:all|element:geometry|color:0x242f3e&style=feature:water|element:geometry|color:0x17263c&markers=color:green|${mapCenter.lat},${mapCenter.lng}&key=AIzaSyAhU4dNUYF6uCCelKWycC9wr0ualDFXYV8`;
    
    return (
      <div className="google-map-container">
        <div className="static-map-fallback">
          <img 
            src={staticMapUrl} 
            alt="Static Map" 
            className="static-map-image"
            onError={() => setUseStaticMap(false)}
          />
          <div className="static-map-overlay">
            <p>Static Map View</p>
            <button 
              className="retry-button"
              onClick={() => {
                setUseStaticMap(false);
                setIsLoading(true);
              }}
            >
              Try Interactive Map
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="google-map-container">
      {locationError && (
        <div className="location-error">
          <p>{locationError}</p>
        </div>
      )}
      
      {/* Swarm Status Indicator */}
      {isSwarmActive && (
        <div className="swarm-status-indicator">
          <div className="swarm-status-content">
            <div className="swarm-status-icon">✈️</div>
            <div className="swarm-status-text">
              <h4>Swarm Coordination Active</h4>
              <p>3 drones coordinating together...</p>
            </div>
          </div>
        </div>
      )}
      <LoadScript
        googleMapsApiKey="AIzaSyAhU4dNUYF6uCCelKWycC9wr0ualDFXYV8"
        libraries={['places']}
        loadingElement={<div className="map-loading">Loading Map...</div>}
        preventGoogleFontsLoading={true}
      >
        <GoogleMap
          mapContainerStyle={mapContainerStyle}
          center={mapCenter}
          zoom={userLocation ? 16 : 15}
          options={mapOptions}
          onLoad={onMapLoad}
        >
          {/* Search Box Overlay */}
          <div className="map-search-container">
            <Autocomplete onLoad={onAutocompleteLoad} onPlaceChanged={onPlaceChanged}>
              <input
                ref={inputRef}
                type="text"
                placeholder="Search address, place, or lat,lng"
                className="map-search-input"
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}
                onKeyDown={onSearchKeyDown}
                autoComplete="off"
              />
            </Autocomplete>
          </div>

          {/* Drone Markers */}
          {drones.map((drone) => (
            <Marker
              key={drone.id}
              position={drone.position}
              icon={getDroneIcon(drone.status, drone.battery)}
              onClick={() => onMarkerClick(drone)}
              title={drone.name}
            />
          ))}

          {/* Swarm Drone Markers */}
          {swarmDrones.map((drone) => (
            <Marker
              key={drone.id}
              position={drone.position}
              icon={{
                path: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z',
                fillColor: drone.color,
                fillOpacity: 1,
                strokeColor: '#ffffff',
                strokeWeight: 2,
                scale: 1.8,
                anchor: (typeof window !== 'undefined' && window.google && window.google.maps)
                  ? new window.google.maps.Point(12, 12)
                  : undefined
              }}
              onClick={() => onMarkerClick(drone)}
              title={drone.name}
            />
          ))}

          {/* Search Result Marker */}
          {searchMarker && (
            <Marker position={searchMarker} />
          )}

          {/* Info Window */}
          {selectedDrone && (
            <InfoWindow
              position={selectedDrone.position}
              onCloseClick={onInfoWindowClose}
            >
              <div className="drone-info-window">
                <h3 style={{ color: '#00bfff', marginBottom: '8px' }}>
                  {selectedDrone.name}
                </h3>
                <div style={{ fontSize: '14px', color: '#e0e7ff' }}>
                  <p><strong>Status:</strong> 
                    <span style={{ 
                      color: selectedDrone.status === 'active' ? '#00ff7f' : '#ff4d4d',
                      marginLeft: '5px'
                    }}>
                      {selectedDrone.status.toUpperCase()}
                    </span>
                  </p>
                  <p><strong>Mission:</strong> {selectedDrone.mission}</p>
                  <p><strong>Battery:</strong> {selectedDrone.battery.toFixed(1)}%</p>
                  <p><strong>Altitude:</strong> {selectedDrone.altitude}m</p>
                  <p><strong>Speed:</strong> {selectedDrone.speed} km/h</p>
                  <p><strong>Last Update:</strong> {selectedDrone.lastUpdate}</p>
                </div>
              </div>
            </InfoWindow>
          )}

          {/* Flight Paths */}
          {flightPaths.map((path, index) => (
            <Polyline
              key={index}
              path={path.coordinates}
              options={{
                strokeColor: path.color || '#00bfff',
                strokeOpacity: 0.8,
                strokeWeight: 3,
                geodesic: true
              }}
            />
          ))}

          {/* Swarm Route */}
          {swarmRoute && (
            <Polyline
              path={swarmRoute.coordinates}
              options={{
                strokeColor: swarmRoute.color,
                strokeOpacity: 0.9,
                strokeWeight: 4,
                geodesic: true
              }}
            />
          )}
        </GoogleMap>
      </LoadScript>

    </div>
  );
};

export default GoogleMapComponent;
