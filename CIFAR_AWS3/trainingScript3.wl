(* ::Package:: *)

(* ::Title:: *)
(*CIFAR-10 AWS Script*)


(* ::Subsection:: *)
(*Data import*)


dir := Directory[]


CloudConnect["laurensorensen@college.harvard.edu", "Chumble1"];


cifar = ResourceObject["CIFAR-10"];
train = ResourceData[cifar, "TrainingData"];
test = ResourceData[cifar, "TestData"];


trainRand = RandomSample[train];
trainingSet = trainRand[[ ;; 45000]];
validationSet = trainRand[[45001 ;; ]];


classes = Union@Values[train]


checkpointsPath = 
	Table[FileNameJoin[{Directory[],"checkpoints/net"}]<>IntegerString[i], {i, 1, 10}]


(* Defining a log to record weights, gradients and batch loss at training checkpoints. *)
logFile5 = OpenWrite[];
logFile7 = OpenWrite[];
logFile8 = OpenWrite[];
logFile9 = OpenWrite[];
logFile10 = OpenWrite[];


cmPath = 
	Table[FileNameJoin[{Directory[],"classifier_measurements/net"}]<>IntegerString[i], {i, 1, 10}]


logPath = 
	Table[FileNameJoin[{Directory[],"training_logs/net"}]<>IntegerString[i], {i, 1, 10}]


lenet = NetChain[{
			ConvolutionLayer[20, 5], 
			Ramp, 
			PoolingLayer[2, 2], 
			ConvolutionLayer[50, 5], 
			Ramp, 
			PoolingLayer[2, 2], 
			FlattenLayer[], 
			500, 
			Ramp, 
			10, SoftmaxLayer[]
		}, 
		"Output" -> NetDecoder[{"Class", classes}], 
		"Input" -> NetEncoder[{"Image", {32, 32}}] 
	]


weightLayers = {1,4,8,10}


appendToLog[n_] := PutAppend[ 
					<|"Weights" -> #Weights,
					"Gradients" -> #Gradients, 
					"BatchLoss" -> #BatchLoss,
					"ValidationLoss" -> #ValidationLoss|>, 
				logFiles[[n]] ]&


appendToLog5 := PutAppend[ 
					<|"Weights" -> #Weights,
					"Gradients" -> #Gradients, 
					"BatchLoss" -> #BatchLoss,
					"ValidationLoss" -> #ValidationLoss|>, 
				logFile5 ]&


appendToLog7 := PutAppend[ 
					<|"Weights" -> #Weights,
					"Gradients" -> #Gradients, 
					"BatchLoss" -> #BatchLoss,
					"ValidationLoss" -> #ValidationLoss|>, 
				logFile7 ]&


appendToLog8 := PutAppend[ 
					<|"Weights" -> #Weights,
					"Gradients" -> #Gradients, 
					"BatchLoss" -> #BatchLoss,
					"ValidationLoss" -> #ValidationLoss|>, 
				logFile8 ]&


(* ::Code::Initialization::Bold:: *)
appendToLog9 := PutAppend[ 
					<|"Weights" -> #Weights,
					"Gradients" -> #Gradients, 
					"BatchLoss" -> #BatchLoss,
					"ValidationLoss" -> #ValidationLoss|>, 
				logFile9 ]&


appendToLog10 := PutAppend[ 
					<|"Weights" -> #Weights,
					"Gradients" -> #Gradients, 
					"BatchLoss" -> #BatchLoss,
					"ValidationLoss" -> #ValidationLoss|>, 
				logFile10 ]&


net = NetInitialize[
		Fold[
			NetReplacePart[#,{#2,"Weights"} -> Automatic]&,
			lenet,
			weightLayers
		]
	]


(* Freeze 1st linear layer. *)
net5 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> Scaled[0.1],
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog5, "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[5]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{8 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[5]], "net5.wlnet"}, net5];


Export[ FileNameJoin@{cmPath[[5]], "net5.mx"}, ClassifierMeasurements[ net5, validationSet ]];


(* Reading the checkpointing log. *)
log5 = ReadList[logFile5];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[6]], "net5.mx"}, log5];


(* Freeze all but 1st conv. layer. *)
net7 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> validationSet,
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog7, "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[7]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{4 -> 0, 8 -> 0, 10 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[7]], "net7.wlnet"}, net7];


Export[ FileNameJoin@{cmPath[[7]], "net7.mx"}, ClassifierMeasurements[ net7, validationSet ]];


(* Reading the checkpointing log. *)
log7 = ReadList[logFile7];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[7]], "net7.mx"}, log7];


(* Freeze all but 2nd conv. layer *)
net8 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> validationSet,
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog8, "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[8]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{1 -> 0, 8 -> 0, 10 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[8]], "net8.wlnet"}, net8];


Export[ FileNameJoin@{cmPath[[8]], "net8.mx"}, ClassifierMeasurements[ net8, validationSet ]];


(* Reading the checkpointing log. *)
log8 = ReadList[logFile8];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[8]], "net8.mx"}, log8];


(* Freeze all but 1st linear layer *)
net9 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> validationSet,
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog9, "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[9]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{1 -> 0, 4 -> 0, 10 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[9]], "net9.wlnet"}, net9];


Export[ FileNameJoin@{cmPath[[9]], "net9.mx"}, ClassifierMeasurements[ net9, validationSet ]];


(* Reading the checkpointing log. *)
log9 = ReadList[logFile9];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[9]], "net9.mx"}, log9];


(* Freeze all but 2nd linear layer *)
net10 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> validationSet,
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog10, "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[10]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{1 -> 0, 4 -> 0, 8 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[10]], "net10.wlnet"}, net10];


Export[ FileNameJoin@{cmPath[[10]], "net10.mx"}, ClassifierMeasurements[ net10, validationSet ]];


(* Reading the checkpointing log. *)
log10 = ReadList[logFile10];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[10]], "net10.mx"}, log10];
