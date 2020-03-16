import { DigitRecognition } from "./digit-recognition/digit-recognition";

export * from "./network/index";
export * from "./digit-recognition/digit-recognition";

const recognizer = new DigitRecognition(100, 0.3, 0, true);
recognizer.train(100, 3);

recognizer.autoTest(1).then((res) => {
	for (const r of res) {
		console.log("Digit: ", r.digit, "TotalPercent: ", r.totalPercent);
		
		for (const i of r.results) {
			console.log("\t", i);
		}
	}
});