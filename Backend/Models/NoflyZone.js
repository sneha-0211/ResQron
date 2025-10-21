// models/NoFlyZone.js
import mongoose from 'mongoose';
const Schema = mongoose.Schema;
const NoFlyZoneSchema = new Schema({
  name: String,
  polygon: { type: Array, required: true }, // array of [lat,lng]
  active: { type: Boolean, default: true }
});
export default mongoose.model('NoFlyZone', NoFlyZoneSchema);
