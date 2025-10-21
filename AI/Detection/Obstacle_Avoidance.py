# realistic_drone_prototype.py
# Run: python realistic_drone_prototype.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import time
from enum import Enum, auto

# ======= CONFIG - tune for prototype =======
DT = 0.1                     # simulation timestep (s)
WORLD_SIZE = 40
MAX_SPEED = 2.0              # m/s horizontal
CRUISE_ALT = 8.0             # safe cruise altitude (m)
LAND_DESCENT_RATE = 0.5      # m per second during landing
CLIMB_RATE = 1.0             # m per second for abort climb
GOAL_RADIUS = 0.8            # m to consider "at goal"
DRONE_RADIUS = 0.5           # horizontal footprint (m)
LIDAR_RANGE = 8.0            # forward sensing (m)
NUM_LIDAR_RAYS = 31
LIDAR_RES_STEP = 0.1         # ray marching step (m)
OBSTACLE_INFLUENCE = 4.0
REPULSION_GAIN = 3.0
ATTRACTION_GAIN = 1.2
DOWNWARD_CLEARANCE_MARGIN = 0.2   # extra margin for landing clearance (m)
LOG_CSV = "telemetry_log.csv"

# ======= State machine =======
class State(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    NAVIGATE = auto()
    LANDING = auto()
    EMERGENCY = auto()
    LANDED = auto()

# ======= Obstacle class with height =======
class Obstacle:
    def __init__(self, x, y, radius, height=6.0):
        self.x = float(x); self.y = float(y); self.r = float(radius); self.h = float(height)

    def horizontal_dist(self, px, py):
        return np.linalg.norm([px - self.x, py - self.y]) - self.r

    def overlaps_xy(self, px, py):
        return np.linalg.norm([px - self.x, py - self.y]) <= (self.r + DRONE_RADIUS)

    def blocks_altitude(self, altitude):
        return altitude <= self.h + DOWNWARD_CLEARANCE_MARGIN

# ======= Drone class (physics + sensors + state machine) =======
class Drone:
    def __init__(self, start_xy, start_alt=0.0):
        self.pos = np.array(start_xy, dtype=float)   # x,y
        self.vel = np.zeros(2)                       # vx, vy
        self.alt = float(start_alt)                  # altitude (m)
        self.state = State.IDLE
        self.heading = 0.0
        self.landed = False
        self.telemetry = []  # list of dicts for logging

    # ---------- Sensor stubs (replace with real sensors) ----------
    def lidar_scan(self, obstacles):
        """
        Simulate a forward fan LIDAR in drone heading direction.
        Returns angles (global) and distances (<= LIDAR_RANGE)
        """
        # fan from -90 deg to +90 deg in body frame
        angles_body = np.linspace(-np.pi/2, np.pi/2, NUM_LIDAR_RAYS)
        angles_world = angles_body + self.heading
        dists = np.full_like(angles_world, LIDAR_RANGE)
        for i, a in enumerate(angles_world):
            ray_dir = np.array([np.cos(a), np.sin(a)])
            # march along ray
            steps = int(LIDAR_RANGE / LIDAR_RES_STEP)
            for s in range(1, steps+1):
                p = self.pos + ray_dir * (s * LIDAR_RES_STEP)
                # check world bounds
                if not (0 <= p[0] <= WORLD_SIZE and 0 <= p[1] <= WORLD_SIZE):
                    dists[i] = s * LIDAR_RES_STEP
                    break
                hit = False
                for obs in obstacles:
                    if obs.horizontal_dist(p[0], p[1]) <= 0:
                        dists[i] = s * LIDAR_RES_STEP
                        hit = True
                        break
                if hit: break
        return angles_world, dists

    def downward_range(self, obstacles):
        """
        Simulate a downward rangefinder (e.g., ultrasonic/LiDAR) returning
        distance to nearest obstacle surface below or ground (assuming ground z=0).
        For prototype, if there's any obstacle whose footprint overlaps XY and whose top
        is above ground, return (altitude - top_of_obstacle) if flying above, else 0.
        We'll return "clear" boolean and measured distance to ground/obstacle surface.
        """
        # check if any obstacle is directly under and high enough to block landing
        min_gap = self.alt  # default distance to ground
        blocked = False
        for obs in obstacles:
            if obs.overlaps_xy(self.pos[0], self.pos[1]):
                top = obs.h
                gap = self.alt - top
                if gap <= 0:
                    # we're inside obstacle - immediate collision (bad)
                    min_gap = -abs(gap)
                    blocked = True
                    break
                else:
                    min_gap = min(min_gap, gap)
                    if top + DOWNWARD_CLEARANCE_MARGIN >= self.alt:
                        blocked = True
        # also measure distance to ground
        ground_gap = self.alt
        min_gap = min(min_gap, ground_gap)
        return (not blocked), min_gap  # (is_clear_to_land, distance_to_surface)

    # ---------- High-level control ----------
    def attractive_velocity(self, goal):
        vec = np.array(goal) - self.pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return np.zeros(2)
        desired = (vec / dist) * ATTRACTION_GAIN * min(dist, MAX_SPEED)
        return desired

    def repulsive_from_obstacles(self, obstacles):
        total = np.zeros(2)
        for obs in obstacles:
            d = obs.horizontal_dist(self.pos[0], self.pos[1])
            if d < OBSTACLE_INFLUENCE:
                dir_away = (self.pos - np.array([obs.x, obs.y]))
                norm = np.linalg.norm(dir_away) + 1e-6
                dir_away /= norm
                strength = REPULSION_GAIN * (1.0/(d+1e-3) - 1.0/OBSTACLE_INFLUENCE)
                if strength > 0:
                    total += dir_away * strength
        return total

    def compute_nominal_velocity(self, goal, obstacles):
        v_att = self.attractive_velocity(goal)
        v_rep = self.repulsive_from_obstacles(obstacles)
        desired = v_att + v_rep
        speed = np.linalg.norm(desired)
        if speed > MAX_SPEED and speed > 1e-6:
            desired = desired / speed * MAX_SPEED
        return desired

    # ---------- State transitions & dynamics ----------
    def update(self, goal, obstacles):
        """
        Call this every DT. Handles state machine and physics update.
        """
        # log base telemetry
        t0 = time.time()
        # State machine
        if self.state == State.IDLE:
            # start takeoff immediately for prototype
            self.state = State.TAKEOFF

        elif self.state == State.TAKEOFF:
            # climb to cruise altitude
            if self.alt < CRUISE_ALT:
                self.alt = min(CRUISE_ALT, self.alt + CLIMB_RATE * DT)
            else:
                self.state = State.NAVIGATE

        elif self.state == State.NAVIGATE:
            # normal navigate towards goal
            desired = self.compute_nominal_velocity(goal, obstacles)
            self.vel = desired
            self.pos += self.vel * DT
            if np.linalg.norm(self.pos - np.array(goal)) <= GOAL_RADIUS:
                # ready to land, but check downward sensor
                clear, gap = self.downward_range(obstacles)
                if clear and gap >= 0:
                    self.state = State.LANDING
                else:
                    # attempt to reposition slightly to find clear patch
                    # simple strategy: random lateral jitter + repulsion to find clear spot
                    jitter = np.random.normal(scale=0.3, size=2)
                    self.pos += jitter * DT

            # detect imminent collision from LIDAR -> emergency
            ang, dists = self.lidar_scan(obstacles)
            if np.any(dists < 0.8):  # threshold close obstacle
                self.state = State.EMERGENCY

        elif self.state == State.LANDING:
            # descent while continuously checking downward sensor
            clear, gap = self.downward_range(obstacles)
            if not clear:
                # abort landing: climb to safe altitude and re-enter NAVIGATE
                self.state = State.EMERGENCY
            else:
                # descend
                self.alt = max(0.0, self.alt - LAND_DESCENT_RATE * DT)
                # small XY corrections to stay centered on goal
                desired = self.attractive_velocity(goal)
                # slow horizontal speed for landing
                self.pos += desired * DT * 0.3
                if self.alt <= 0.01:
                    self.alt = 0.0
                    self.landed = True
                    self.state = State.LANDED

        elif self.state == State.EMERGENCY:
            # simple emergency behavior:
            # - if too low, climb quickly to CRUISE_ALT + 1 and continue navigating
            if self.alt < CRUISE_ALT + 1.0:
                self.alt = min(CRUISE_ALT + 1.0, self.alt + CLIMB_RATE * DT)
            else:
                # after climb, go back to NAVIGATE and try to find other approach
                self.state = State.NAVIGATE

        elif self.state == State.LANDED:
            self.landed = True
            self.vel = np.zeros(2)

        # update heading
        if np.linalg.norm(self.vel) > 1e-6:
            self.heading = np.arctan2(self.vel[1], self.vel[0])

        # telemetry record
        self.telemetry.append({
            "t": time.time(),
            "state": self.state.name,
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "alt": float(self.alt),
            "vx": float(self.vel[0]),
            "vy": float(self.vel[1])
        })

    # ---------- logging ----------
    def dump_log(self, filename=LOG_CSV):
        df = pd.DataFrame(self.telemetry)
        df.to_csv(filename, index=False)
        print(f"[LOG] Telemetry saved to {filename}")


# ======= World creation helper =======
def create_obstacles():
    # Example obstacles: (x,y,radius,height)
    return [
        Obstacle(12, 12, 3.0, height=5.0),
        Obstacle(20, 6, 2.5, height=7.0),
        Obstacle(24, 18, 3.0, height=4.0),
        Obstacle(30, 25, 2.0, height=6.0),
        # narrow corridor
        Obstacle(16, 16, 1.2, height=3.5),
        Obstacle(18, 16, 1.2, height=3.5),
    ]


# ======= Visualization & run loop =======
def run_simulation(start, goal, obstacles, max_steps=2000):
    drone = Drone(start, start_alt=0.0)
    drone.state = State.IDLE

    # Prepare plot
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_xlim(0, WORLD_SIZE); ax.set_ylim(0, WORLD_SIZE); ax.set_aspect('equal')
    ax.set_title("Software Prototype: Drone Obstacle Avoidance + Safe Landing (2.5D)")

    # Draw obstacles
    patches = []
    for obs in obstacles:
        c = plt.Circle((obs.x, obs.y), obs.r, color='gray', alpha=0.7)
        ax.add_patch(c)
        patches.append(c)

    # Drone marker: draw pos + circle for footprint + text altitude
    drone_dot, = ax.plot([], [], marker=(3,0,0), markersize=10)
    footprint = plt.Circle((0,0), DRONE_RADIUS, color='blue', alpha=0.2)
    ax.add_patch(footprint)
    goal_dot, = ax.plot(goal[0], goal[1], marker='*', markersize=14, color='gold')
    path_line, = ax.plot([], [], linestyle='--', lw=1)
    status_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va='top')

    lidar_lines = [ax.plot([], [], lw=1, alpha=0.4)[0] for _ in range(NUM_LIDAR_RAYS)]

    path_x, path_y = [], []

    step = {'i': 0}

    def init():
        drone_dot.set_data([], [])
        footprint.center = (0,0)
        path_line.set_data([], [])
        status_text.set_text('')
        for ln in lidar_lines:
            ln.set_data([], [])
        return [drone_dot, footprint, path_line, status_text] + lidar_lines

    def animate(frame):
        step['i'] += 1
        drone.update(goal, obstacles)

        # update visuals
        path_x.append(drone.pos[0]); path_y.append(drone.pos[1])
        drone_dot.set_data([drone.pos[0]], [drone.pos[1]])
        footprint.center = (drone.pos[0], drone.pos[1])
        path_line.set_data(path_x, path_y)
        status_text.set_text(f"State: {drone.state.name} | Pos: ({drone.pos[0]:.1f},{drone.pos[1]:.1f}) | Alt: {drone.alt:.1f}")

        # lidar rays
        angs, dists = drone.lidar_scan(obstacles)
        for i, ln in enumerate(lidar_lines):
            a = angs[i]
            d = dists[i]
            x2 = drone.pos[0] + np.cos(a) * d
            y2 = drone.pos[1] + np.sin(a) * d
            ln.set_data([drone.pos[0], x2], [drone.pos[1], y2])
        # stop conditions
        if drone.state == State.LANDED or step['i'] >= max_steps:
            anim.event_source.stop()
            drone.dump_log()
        return [drone_dot, footprint, path_line, status_text] + lidar_lines

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=2000, interval=DT*1000, blit=True)
    plt.show()
    return drone

# ======= Extension points / integration stubs =======
def mavlink_send_command_stub(cmd):
    """
    Stub: replace with real MAVLink call to send velocity/position or mode changes.
    For prototype testing, we keep it offline.
    """
    # Example: print(cmd)
    pass

def ros_publish_stub(topic, data):
    """
    Stub: replace with ROS publisher to send telemetry or receive sensor data.
    """
    pass

# ======= MAIN =======
if __name__ == "__main__":
    START = (2.0, 2.0)
    GOAL = (34.0, 28.0)

    obstacles = create_obstacles()
    drone = run_simulation(START, GOAL, obstacles)
    print("Simulation complete.")
]]]