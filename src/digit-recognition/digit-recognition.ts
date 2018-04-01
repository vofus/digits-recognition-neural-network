import { Network, TrainSet, ITrainItem } from "../network";
const mnist = require("mnist");

/**
 * Элемент библиотеки MNIST
 */
interface MnistItem {
	input: number[];
	output: number[];
}

/**
 * Фасад для класса Network,
 * определяющий интерфейс нейронной сети по распознаванию цифр
 */
export class DigitRecognition {
	private static INPUT: number = 784; // количество нейронов во входном слое
	private static OUTPUT: number = 10; // количество нейронов в выходном слое
	private static TRAIN_SET_SIZE: number = 1000; // размер тренировочной выборки
	private nn: Network;

	/**
	 * Создаем тренировочную выборку
	 */
	private static createTrainSet(): TrainSet {
		const {training} = mnist.set(DigitRecognition.TRAIN_SET_SIZE, 0);

		return training.map((trainInem: MnistItem) => ({
			inputs: trainInem.input,
			targets: trainInem.output
		}));
	}

	/**
	 * Constructor
	 * @param hiddenSize размер скрытого слоя
	 * @param LR скорость обучения
	 */
	constructor(hiddenSize: number, LR: number) {
		this.nn = new Network(DigitRecognition.INPUT, hiddenSize, DigitRecognition.OUTPUT, LR);
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
	async autoTest(eachDigitCount: number = 50): Promise<any> {
		throw new Error("Требуется реализация метода autoTest!");
	}

	/**
	 * Тестируем обученную сеть в ручном режиме
	 * и возвращаем результат, ожидаемый результат и процент распознавания
	 * @param digitImg изображение цифры
	 */
	async manualTest(digitImg: any): Promise<any> {
		throw new Error("Требуется реализация метода manualTest!");
	}
}
