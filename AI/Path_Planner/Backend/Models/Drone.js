// models/Drone.js
import mongoose from 'mongoose';
const Schema = mongoose.Schema;

const DroneSchema = new Schema({
  callsign: { type: String, required: true, unique: true },
  type: { type: String, default: 'quadcopter' },
  maxPayloadKg: { type: Number, default: 10 },
  battery: { type: Number, default: 100 }, // percent
  mode: { type: String, default: 'IDLE' },
  lastSeen: { type: Date, default: Date.now },
  location: {
    lat: { type: Number, default: 0 },
    lng: { type: Number, default: 0 },
    alt: { type: Number, default: 0 }
  },
  path: { type: Array, default: [] } // array of [lat,lng] historic points (bounded client-side)
}, { timestamps: true });

export default mongoose.model('Drone', DroneSchema);
