import { Network, TrainSet } from "./network";
import ndarray from "ndarray";
import nj from "numjs";

const trainSet: TrainSet = [
	{ inputs: [1, 0], targets: [1, 0] },
	{ inputs: [0, 1], targets: [0, 1] },
	{ inputs: [0, 0], targets: [0, 0] },
	{ inputs: [1, 1], targets: [0, 0] }
];

const trainSet_02: TrainSet = [
	{ inputs: [1, 0], targets: [1] },
	{ inputs: [0, 1], targets: [1] },
	{ inputs: [0, 0], targets: [0] },
	{ inputs: [1, 1], targets: [0] }
];

const nn = new Network(2, 4, 1, 0.285);

console.time("Train");
// testNN.train(trainSet, 10);
nn.train(trainSet_02, 25000);
console.timeEnd("Train");

(async () => {
	Network.serialize(nn, "test-model.json");
	// const nn: Network = await Network.deserialize("test-model");

	for (const item of trainSet_02) {
		const result = nn.query(item.inputs);
		console.log();
		console.log(result);
		// console.log(`----------- ${result.get(0, 0) >= 0.85 || result.get(0, 1) >= 0.85 ? "TRUE" : "FALSE"} ------------`);
	}
})();