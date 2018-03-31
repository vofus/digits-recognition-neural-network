import { Network, TrainSet } from "./network";
const mnist = require("mnist");

const {training, test} = mnist.set(100, 10);

interface MnistItem {
	input: number[];
	output: number[];
}

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

const MnistTrainSet: TrainSet = training.map((item: MnistItem) => {
	return {
		inputs: item.input,
		targets: item.output
	};
});

const MnistTestSet: TrainSet = test.map((item: MnistItem) => {
	return {
		inputs: item.input,
		targets: item.output
	};
});



const nn = new Network(784, 784, 10, 0.285);

console.time("Train");
nn.train(MnistTrainSet, 1000);
console.timeEnd("Train");

(async () => {
	try {
		// await Network.serialize(nn, "mnist-model.json");
		// const nn: Network = await Network.deserialize("mnist-model.json");

		for (const item of MnistTestSet) {
			const result = nn.query(item.inputs);
			console.log(result);
			console.log("- - - - - - - -");
			console.log(item.targets);
			console.log("============================");
			// console.log(`----------- ${result.get(0, 0) >= 0.85 || result.get(0, 1) >= 0.85 ? "TRUE" : "FALSE"} ------------`);
		}
	} catch (err) {
		console.error(err);
	}
})();