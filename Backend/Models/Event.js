// models/Event.js
import mongoose from 'mongoose';
const Schema = mongoose.Schema;

const EventSchema = new Schema({
  type: String,
  payload: Schema.Types.Mixed,
  source: String,
  createdAt: { type: Date, default: Date.now }
});
export default mongoose.model('Event', EventSchema);
