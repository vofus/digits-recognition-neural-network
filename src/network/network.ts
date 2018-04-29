import fs from "fs";
import { promisify } from "util";
import nj, { NdArray } from "numjs";
import { getFileName, getRandomInt } from "./utils";
import { ActivationStrategy, Sigmoid } from "./activators";
import { shuffle as _shuffle } from "lodash";

const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);

export type ModelNN = { IH: NdArray, HO: NdArray, LR: number, activator: ActivationStrategy };

export interface INetwork {
	train(trainSet: TrainSet, count: number, activator?: ActivationStrategy): void;

	query(inputs: number[]): any;
}

export interface IModel<T> {
	getModel(): T;
}

interface IForwardResult {
	hiddenOutputs: NdArray;
	finalOutputs: NdArray;
}

export interface ITrainItem {
	inputs: number[];
	targets: number[];
}

export type TrainSet = ITrainItem[];

export class Network implements INetwork, IModel<ModelNN> {
	// сериализуем тренированную модель
	static async serialize(nn: Network, filePath: string) {
		const {IH, HO, LR} = nn.getModel();

		try {
			const data = {
				IH: IH.tolist(),
				HO: HO.tolist(),
				LR
			};

			const jsonData = JSON.stringify(data);

			if (!fs.existsSync(filePath)) {
				fs.mkdirSync(filePath);
			}

			await writeFile(getFileName(filePath), jsonData);
		} catch (err) {
			throw err;
		}
	}

	// десериализуем тренированную модель
	static async deserialize(filePath: string): Promise<Network> {
		try {
			const jsonStr: string = await readFile(getFileName(filePath), "utf8");
			const {IH, HO, LR} = JSON.parse(jsonStr);
			const weightsIH = nj.array(IH);
			const weightsHO = nj.array(HO);

			const [hiddenSize, inputSize] = weightsIH.shape;
			const [outputSize] = weightsHO.shape;
			const nn: Network = new Network(inputSize, hiddenSize, outputSize);

			nn.setModel({
				IH: weightsIH,
				HO: weightsHO,
				LR,
				activator: new Sigmoid()
			});

			return nn;
		} catch (err) {
			throw err;
		}
	}

	// Матрица весов между входным и скрытым слоем
	private weightsIH: NdArray;
	// Матрица весов между скрытым и выходным слоем
	private weightsHO: NdArray;
	// Объект-активатор (По умолчанию сигмоида)
	private activator: ActivationStrategy = new Sigmoid();

	/**
	 * Getter модели сети
	 */
	getModel(): ModelNN {
		return {
			IH: this.weightsIH,
			HO: this.weightsHO,
			LR: this.LR,
			activator: this.activator
		};
	}

	/**
	 * Setter модели сети
	 */
	setModel(model: ModelNN): void {
		const {IH, HO, LR, activator} = model;

		this.weightsIH = IH;
		this.weightsHO = HO;
		this.LR = LR;
		this.activator = activator;
	}

	constructor(inputSize: number, hiddenSize: number, outputSize: number, private LR: number = 0.3) {
		this.weightsIH = this.generateWeights(hiddenSize, inputSize);
		this.weightsHO = this.generateWeights(outputSize, hiddenSize);
	}

	/**
	 * Тренируем сеть
	 * @param trainSet {TrainSet} тренировочная выборка
	 * @param epochs {number} количество эпох обучения
	 * @param activator {ActivationStrategy} объект-активатор
	 */
	train(trainSet: TrainSet, epochs: number, activator?: ActivationStrategy): void {
		if (Boolean(activator)) {
			this.activator = activator;
		}

		let epochCounter = epochs;
		while (epochCounter > 0) {
			const shuffled: TrainSet = _shuffle(trainSet);
			let trainCounter = trainSet.length - 1;

			console.time(`Epoch ${epochCounter}`);
			while (trainCounter >= 0) {
				const {inputs, targets} = shuffled[trainCounter];
				this.trainStep(inputs, targets);
				trainCounter -= 1;
			}

			console.timeEnd(`Epoch ${epochCounter}`);
			epochCounter -= 1;
		}
	}

	/**
	 * Выполняем запрос к сети
	 * @param inputs {number[]} входные сигналы
	 */
	query(inputs: number[]): number[] {
		const inputMatrix = nj.array(inputs).reshape(1, inputs.length).T as NdArray;
		const {finalOutputs} = this.forwardPropagation(inputMatrix);

		return finalOutputs.tolist<number[]>().reduce((res, item) => {
			res.push(...item);

			return res;
		}, []);
	}


	/**
	 * Шаг обучения
	 * @param inputs {number[]} входные сигналы
	 * @param targets {number[]} ожидаемый результат
	 */
	private trainStep(inputs: number[], targets: number[]): void {
		const inputMatrix = nj.array(inputs, "float64").reshape(1, inputs.length).T as NdArray;
		const targetMatrix = nj.array(targets, "float64").reshape(1, targets.length).T as NdArray;

		const forwardResult = this.forwardPropagation(inputMatrix);
		this.backPropagation(inputMatrix, targetMatrix, forwardResult);
	}

	/**
	 * Генерируем начаьную матрицу весов
	 * @param rows {number} Количество строк
	 * @param columns {number} Количество столбцов
	 * @returns {NdArray}
	 */
	private generateWeights(rows: number, columns: number): NdArray {
		return nj.random([rows, columns]).subtract(0.5);
	}

	/**
	 * Подсчитываем дополнительные веса
	 * @param inputs {NdArray}
	 * @param outputs {NdArray}
	 * @param errors {NdArray}
	 * @returns {NdArray}
	 */
	private calcAdditionalWeights(inputs: NdArray, outputs: NdArray, errors: NdArray): NdArray {
		const ones = nj.ones(outputs.shape) as NdArray;
		const arg1 = errors.multiply(outputs).multiply(nj.subtract(ones, outputs));
		const arg2 = inputs.T;

		return nj.dot(arg1, arg2).multiply(this.LR);
	}


	/**
	 * Прямое распространение сигнала
	 * @param inputMatrix {NdArray} Входные сигналы, приобразованные в двумерный массив
	 * @returns {IForwardResult}
	 */
	private forwardPropagation(inputMatrix: NdArray): IForwardResult {
		const hiddenInputs = this.weightsIH.dot(inputMatrix);
		const hiddenOutputs = this.activator.execute(hiddenInputs);

		const finalInputs = this.weightsHO.dot(hiddenOutputs);
		const finalOutputs = this.activator.execute(finalInputs);

		return {
			hiddenOutputs,
			finalOutputs
		};
	}


	/**
	 * Обратное распространение ошибки
	 * @param inputMatrix {NdArray} Входные сигналы, приобразованные в двумерный массив
	 * @param targetMatrix {NdArray} Ожидаемые результаты, приобразованные в двумерный массив
	 * @param result {IForwardResult} Объект с выходными сигналами на слоях после прямого прохода
	 */
	private backPropagation(inputMatrix: NdArray, targetMatrix: NdArray, result: IForwardResult): void {
		const {hiddenOutputs, finalOutputs} = result;
		const outputErrors = targetMatrix.subtract(finalOutputs);
		const hiddenErrors = this.weightsHO.T.dot(outputErrors);
		const additionalHO = this.calcAdditionalWeights(hiddenOutputs, finalOutputs, outputErrors);
		const additionalIH = this.calcAdditionalWeights(inputMatrix, hiddenOutputs, hiddenErrors);

		this.weightsHO = this.weightsHO.add(additionalHO);
		this.weightsIH = this.weightsIH.add(additionalIH);
	}
}
