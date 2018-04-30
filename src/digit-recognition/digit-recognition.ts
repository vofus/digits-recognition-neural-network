import { Network, TrainSet, ITrainItem } from "../network";
import _round from "lodash/round";
import _flow from "lodash/fp/flow";
import _map from "lodash/fp/map";
import _filter from "lodash/fp/filter";
import _minBy from "lodash/fp/minBy";
import _maxBy from "lodash/fp/maxBy";
import _getOr from "lodash/fp/getOr";

const mnist = require("mnist");

/**
 * Элемент библиотеки MNIST
 */
interface MnistItem {
	input: number[];
	output: number[];
}

/**
 * Элемент с информацией о распознавании цифры из набора для одной цифры
 */
export interface RecognizedItem {
	isRecognized: boolean;
	recognition: number;
	recognitionPercent: number;
}

/**
 * Информация по тестированию одной цифры
 */
export interface AutoTestDigit {
	digit: number;
	totalRecognition: number;
	minRecognition: number;
	maxRecognition: number;
}

/**
 * Результат, возвращаемый методом автоматического тестирования
 */
export interface AutoTestResult {
	digit: number;
	totalPercent: number;
	results: RecognizedItem[];
}

export interface ManualTestResult {
	digit: number;
	percent: number;
}

/**
 * Фасад для класса Network,
 * определяющий интерфейс нейронной сети по распознаванию цифр
 */
export class DigitRecognition {
	private static INPUT: number = 784; // количество нейронов во входном слое
	private static OUTPUT: number = 10; // количество нейронов в выходном слое
	private static TRAIN_SET_SIZE: number = 10000; // размер тренировочной выборки
	private nn: Network;

	/**
	 * Создаем тренировочную выборку
	 */
	private static createTrainSet(trainSetSize: number = DigitRecognition.TRAIN_SET_SIZE): TrainSet {
		const {training} = mnist.set(trainSetSize, 0);

		return DigitRecognition.prepareMnistItems(training);
	}

	/**
	 * Преобразуем данные к виду, принимаемому нашей сетью
	 */
	private static prepareMnistItems(mnistItems: any): TrainSet {
		return mnistItems.map((trainInem: MnistItem) => ({
			inputs: trainInem.input,
			targets: trainInem.output
		}));
	}

	/**
	 * Находим индекс максимального числа в массиве
	 */
	private static getMaxIndex(arr: number[]): number {
		if (!Array.isArray(arr)) {
			throw new Error("Method getMaxIndex expected Array!");
		}

		if (arr.length === 0) {
			return undefined;
		}

		if (arr.length === 1) {
			return 0;
		}

		let index = 0;
		for (let i = 1; i < arr.length; ++i) {
			index = arr[i] > arr[i - 1] ? i : index;
		}

		return index;
	}

	/**
	 * Constructor
	 * Принимает объект Network,
	 * либо размер скрытого слоя и коэффициент скорости обучения
	 */
	constructor(network: Network)
	constructor(hiddenSize: number, LR: number)
	constructor(hiddenOrNetwork: number | Network, LR?: number) {
		if (typeof hiddenOrNetwork === "object" && hiddenOrNetwork instanceof Network) {
			this.nn = hiddenOrNetwork;
		} else if (typeof hiddenOrNetwork === "number" && typeof LR === "number") {
			this.nn = new Network(DigitRecognition.INPUT, hiddenOrNetwork, DigitRecognition.OUTPUT, LR);
		}
	}

	/**
	 * Network getter
	 */
	get network() {
		return this.nn;
	}

	/**
	 * Создаем и тренируем сеть
	 * @param epochs количество эпох обучения
	 */
	async train(trainSetSize: number, epochs: number): Promise<void> {
		try {
			const trainSet: TrainSet = DigitRecognition.createTrainSet(trainSetSize);
			this.nn.train(trainSet, epochs);
		} catch (err) {
			throw err;
		}
	}

	/**
	 * Тестируем обученную сеть в автоматическом режиме
	 * и возвращаем отчет по результатам тестирования
	 * @param eachDigitCount количество тестов для каждой цифры
	 */
	async autoTest(eachDigitCount: number = 50): Promise<AutoTestResult[]> {
		const results: AutoTestResult[] = [];

		for (let i = 0; i < 10; ++i) {
			const testSet = DigitRecognition.prepareMnistItems(mnist[i].set(0, eachDigitCount));

			const items: RecognizedItem[] = _map((item: ITrainItem): RecognizedItem => {
				const result = this.nn.query(item.inputs);
				const maxIndex = DigitRecognition.getMaxIndex(result);

				return {
					isRecognized: maxIndex === i,
					recognition: maxIndex,
					recognitionPercent: _round(result[maxIndex], 2)
				};
			})(testSet);

			const recognized: RecognizedItem[] = _filter<RecognizedItem>((item) => item.isRecognized)(items);

			results.push({
				digit: i,
				totalPercent: _round(recognized.length / eachDigitCount, 2),
				results: items
			});
		}

		return results;
	}

	/**
	 * Тестируем обученную сеть в ручном режиме
	 * и возвращаем результат, ожидаемый результат и процент распознавания
	 * @param inputs изображение цифры
	 */
	async manualTest(inputs: number[]): Promise<ManualTestResult> {
		const result = this.nn.query(inputs);
		const maxIndex = DigitRecognition.getMaxIndex(result);

		return {
			digit: maxIndex,
			percent: _round(result[maxIndex], 2)
		};
	}
}
