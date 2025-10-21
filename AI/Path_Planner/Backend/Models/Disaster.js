import mongoose from 'mongoose';
const Schema = mongoose.Schema;

const DisasterSchema = new Schema({
  type: { type: String, required: true }, // flood, fire, earthquake, landslide, other
  severity: { type: String, required: true }, // low, moderate, high, critical
  confidence: { type: Number, required: true }, // 0-1
  coordinates: { 
    lat: { type: Number, required: true },
    lng: { type: Number, required: true }
  },
  description: { type: String, required: true },
  recommendedActions: [{ type: String }],
  status: { type: String, default: 'detected' }, // detected, investigating, responding, resolved
  assignedDrones: [{ type: String }], // array of callsigns
  imageUrl: { type: String },
  detectedAt: { type: Date, default: Date.now },
  resolvedAt: { type: Date },
  metadata: { type: Schema.Types.Mixed }
}, { timestamps: true });

export default mongoose.model('Disaster', DisasterSchema);
