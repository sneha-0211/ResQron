@echo off
REM 

echo Starting ResQron AI Services...

REM 
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js not found. Please install Node.js 16+
    pause
    exit /b 1
)

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker not found. MQTT and MongoDB services will not start automatically.
    echo Please start them manually or install Docker.
) else (
    echo Starting infrastructure services with Docker...
    
    REM Start MQTT broker
    docker run -d --name resqron-mqtt -p 1883:1883 eclipse-mosquitto:2.0
    if errorlevel 1 (
        echo MQTT broker may already be running
    )
    
    REM Start MongoDB
    docker run -d --name resqron-mongodb -p 27017:27017 mongo:7.0
    if errorlevel 1 (
        echo MongoDB may already be running
    )
    
    echo Waiting for services to start...
    timeout /t 5 /nobreak >nul
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r ai/requirements.txt
if errorlevel 1 (
    echo Failed to install Python dependencies
    pause
    exit /b 1
)

REM Install Node.js dependencies
echo Installing Node.js dependencies...
cd backend
npm install
if errorlevel 1 (
    echo Failed to install backend dependencies
    pause
    exit /b 1
)

cd ../dashboard
npm install
if errorlevel 1 (
    echo Failed to install dashboard dependencies
    pause
    exit /b 1
)

cd ..

REM Start AI services using the integration script
echo Starting AI services...
python integration_setup.py --start-all
if errorlevel 1 (
    echo Failed to start AI services
    pause
    exit /b 1
)

echo.
echo All services started successfully!
echo.
echo Service URLs:
echo   Dashboard: http://localhost:3000
echo   Backend API: http://localhost:5000
echo   Planner API: http://localhost:8000
echo   Depth API: http://localhost:8001
echo.
echo Press any key to stop all services...
pause >nul

echo Stopping all services...
python integration_setup.py --stop-all

REM Stop Docker containers
docker stop resqron-mqtt resqron-mongodb >nul 2>&1
docker rm resqron-mqtt resqron-mongodb >nul 2>&1

echo All services stopped.
pause
