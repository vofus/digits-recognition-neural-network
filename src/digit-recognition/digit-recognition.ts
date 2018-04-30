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
	testsCount: number;
	digits: any[];
}

export interface ManualTestResult {
	digit: number;
	percent: number;
}

/**
 * Элемент с информацией о распознавании цифры из набора для одной цифры
 */
interface RecognizedItem {
	isRecognized: boolean;
	recognition: number;
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
	private static createTrainSet(): TrainSet {
		const {training} = mnist.set(DigitRecognition.TRAIN_SET_SIZE, 0);

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
	async train(epochs: number): Promise<void> {
		try {
			const trainSet: TrainSet = DigitRecognition.createTrainSet();
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
	async autoTest(eachDigitCount: number = 50): Promise<AutoTestResult> {
		const results: AutoTestDigit[] = [];

		for (let i = 0; i < 10; ++i) {
			const testSet = DigitRecognition.prepareMnistItems(mnist[i].set(100, 100 + eachDigitCount));

			const recognized: RecognizedItem[] = _flow([
				_map((item: ITrainItem): RecognizedItem => {
					const result = this.nn.query(item.inputs);
					const maxIndex = DigitRecognition.getMaxIndex(result);

					return {
						isRecognized: maxIndex === i,
						recognition: maxIndex === i ? _round(result[i], 2) : null
					};
				}),
				_filter((item: RecognizedItem) => item.isRecognized)
			])(testSet);

			results.push({
				digit: i,
				totalRecognition: _round(recognized.length / eachDigitCount, 2),
				minRecognition: _flow([_minBy("recognition"), _getOr(0, "recognition")])(recognized),
				maxRecognition: _flow([_maxBy("recognition"), _getOr(0, "recognition")])(recognized),
			});
		}

		return {
			testsCount: eachDigitCount,
			digits: results
		};
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
			percent: result[maxIndex]
		};
	}
}
