import { DigitRecognition } from "./digit-recognition/digit-recognition";

export * from "./network/index";
export * from "./digit-recognition/digit-recognition";

const recognizer = new DigitRecognition(100, 0.3, 1, true);