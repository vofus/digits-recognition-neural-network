import { Network, TrainSet } from "./network";
import { DigitRecognition } from "./digit-recognition/digit-recognition";
import path from "path";
const mnist = require("mnist");

const {test} = mnist.set(0, 10);
const MODELS_PATH: string = path.resolve(__dirname, "../models");

interface MnistItem {
	input: number[];
	output: number[];
}

const MnistTestSet: TrainSet = test.map((item: MnistItem) => {
	return {
		inputs: item.input,
		targets: item.output
	};
});


// const nn = new Network(784, 200, 10, 0.085);

// console.time("Train");
// nn.train(MnistTrainSet, 50);
// console.timeEnd("Train");

(async () => {
	try {
		const nn = new DigitRecognition(15, 0.117);
		await nn.train(100);
		// const nn: Network = await DigitRecognition.create(15, 100, 0.117);
		// await Network.serialize(nn, MODELS_PATH, "mnist-model-03.json");
		// const nn: Network = await Network.deserialize(MODELS_PATH, "mnist-model-02.json");

		// for (const item of MnistTestSet) {
		// 	const result = nn.query(item.inputs);
		// 	console.log(result);
		// 	console.log("- - - - - - - -");
		// 	console.log(item.targets);
		// 	console.log("============================");
		// 	// console.log(`----------- ${result.get(0, 0) >= 0.85 || result.get(0, 1) >= 0.85 ? "TRUE" : "FALSE"} ------------`);
		// }
	} catch (err) {
		console.error(err);
	}
})();