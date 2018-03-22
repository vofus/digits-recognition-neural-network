import { Network, TrainSet } from "./network";
import ndarray from "ndarray";
import nj from "numjs";

const trainSet: TrainSet = [
	{ inputs: [1, 0], targets: [1, 0] },
	{ inputs: [0, 1], targets: [0, 1] },
	{ inputs: [0, 0], targets: [0, 0] },
	{ inputs: [1, 1], targets: [0, 0] }
];

const testNN = new Network(2, 8, 2, 0.0185);

console.time("Train");
testNN.train(trainSet, 50000);
console.timeEnd("Train");

for (const item of trainSet) {
	const result = testNN.query(item.inputs);
	console.log();
	console.log(result);
	console.log(`----------- ${result.get(0, 0) >= 0.85 || result.get(0, 1) >= 0.85 ? "TRUE" : "FALSE"} ------------`);
}