// models/Mission.js
import mongoose from 'mongoose';
const Schema = mongoose.Schema;

const MissionSchema = new Schema({
  name: { type: String, default: '' },
  waypoints: { type: Array, required: true }, // array of [lat,lng]
  supplies: { type: Array, default: [] },
  priority: { type: Number, default: 5 }, // 1 highest
  assignedTo: { type: String, default: null }, // callsign
  status: { type: String, default: 'queued' }, // queued, active, completed, failed
  createdAt: { type: Date, default: Date.now },
  metadata: { type: Schema.Types.Mixed }
});

export default mongoose.model('Mission', MissionSchema);
