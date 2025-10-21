import mongoose from 'mongoose';
import dotenv from 'dotenv';
import Drone from '../models/Drone.js';
dotenv.config();
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/sih';

await mongoose.connect(MONGO_URI);
console.log('Mongo connected for seed');

const list = [
  { callsign: 'DRONE_A', type: 'heavy', maxPayloadKg: 10, battery: 100, location:{lat:28.600, lng:77.200} },
  { callsign: 'DRONE_B', type: 'heavy', maxPayloadKg: 10, battery: 100, location:{lat:28.605, lng:77.205} }
];

for (const d of list) {
  await Drone.updateOne({ callsign:d.callsign }, { $set:d }, { upsert:true });
  console.log('seeded', d.callsign);
}
process.exit(0);
