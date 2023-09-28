from .StdGP import StdGP
from sklearn.preprocessing import MinMaxScaler
from random import Random
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-rbdGP
#
# Copyright ©2023-2023 J. E. Batista
#

import warnings
warnings.filterwarnings("ignore")

class ClassifierNotTrainedError(Exception):
    """ You tried to use the classifier before training it. """

    def __init__(self, expression, message = ""):
        self.expression = expression
        self.message = message


class RbdGP:

	## __INIT__ arguments
	operators = None
	max_initial_depth = None
	population_size = None
	threads = None
	random_state = 42
	rng = None # random number generator

	max_depth = None
	max_generation = None
	tournament_size = None
	elitism_size = None

	model_name = None 
	fitnessType = None

	verbose = None


	## FIT arguments

	trainingAccuracyOverTime = None
	testAccuracyOverTime = None
	trainingWaFOverTime = None
	testWaFOverTime = None
	trainingKappaOverTime = None
	testKappaOverTime = None
	trainingMSEOverTime = None
	testMSEOverTime = None
	sizeOverTime = None
	generationTimes = None

	final_models = None

	trained = False


	def checkIfTrained(self):
		if self.trained:
			raise ClassifierNotTrainedError("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")



	def __init__(self, operators=[("+",2),("-",2),("*",2),("/",2)], max_initial_depth = 6, population_size = 200, 
		max_generation = 100, tournament_size = 5, elitism_size = 1, max_depth = 17, 
		threads=1, random_state = 42, verbose = True, model_name="SimpleThresholdClassifier", fitnessType="Accuracy"):

		if sum( [0 if op in [("+",2),("-",2),("*",2),("/",2)] else 0 for op in operators ] ) > 0:
			print( "[Warning] Some of the following operators may not be supported:", operators)

		self.operators = operators

		self.max_initial_depth = max_initial_depth
		self.population_size = population_size
		self.threads = max(1, threads)
		self.random_state = random_state
		self.rng = Random(random_state)

		self.max_depth = max_depth
		self.max_generation = max_generation
		self.tournament_size = tournament_size
		self.elitism_size = elitism_size

		self.model_name = model_name
		self.fitnessType = fitnessType

		self.verbose = verbose



	def __str__(self):
		self.checkIfTrained()
		return ",".join([str(d) for d in self.final_dimensions])
		



	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingAccuracyOverTime, self.testAccuracyOverTime]
	
	def getFitnessOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		self.checkIfTrained()

		return self.fitnessOverTime

	def getWaFOverTime(self):
		'''
		Returns the training and test WAF of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingWaFOverTime, self.testWaFOverTime]

	def getKappaOverTime(self):
		'''
		Returns the training and test kappa values of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingKappaOverTime, self.testKappaOverTime]

	def getMSEOverTime(self):
		'''
		Returns the training and test mean squared error values of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingMSEOverTime, self.testMSEOverTime]

	def getSizesOverTime(self):
		'''
		Returns the size and number of dimensions of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.sizeOverTime, self.dimensions]

	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.generationTimes

	def getBestIndividual(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return ",".join([str(s) for s in self.final_models])
	

	def getPopulationsOverTime(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.modelos









	def fit(self,Tr_x, Tr_y, Te_x = None, Te_y = None):
		if self.verbose:
			print("  > Parameters")
			print("    > Random State:       "+str(self.random_state))
			print("    > Operators:          "+str(self.operators))
			print("    > Population Size:    "+str(self.population_size))
			print("    > Max Generation:     "+str(self.max_generation))
			print("    > Tournament Size:    "+str(self.tournament_size))
			print("    > Elitism Size:       "+str(self.elitism_size))
			print("    > Max Initial Depth:  "+str(self.max_initial_depth))
			print("    > Max Depth:          "+str(self.max_depth))
			print("    > Wrapped Model:      "+self.model_name)
			print("    > Fitness Type:       "+self.fitnessType)
			print("    > Threads:            "+str(self.threads))
			print()

		self.Tr_x = Tr_x
		self.Tr_y = Tr_y
		self.Te_x = Te_x
		self.Te_y = Te_y
		self.terminals = list(Tr_x.columns)


		self.sizeOverTime = [0]*self.max_generation
		self.generationTimes = [0]*self.max_generation
		self.fitnessOverTime = [0]*self.max_generation
		self.modelos = []

		classes = list(set(list(self.Tr_y)))
		n_classes = len(classes)
		n_runs = min(50, 3*n_classes)

		self.dimensions = [n_runs] * self.max_generation
		models_tmp = []

		for r in range(n_runs):
			model =	StdGP(self.operators, self.max_depth, self.population_size, 
				self.max_generation, self.tournament_size, self.elitism_size, 
				self.max_depth, self.threads, r, True, self.model_name, self.fitnessType)
			
			# Fazer uma conversao random de Tr_y e Te_y para o problema ser binario
			# eg: classe random em 0, classe random em 1, distribuir o resto aleatoriamente

			classes_dup = classes[:]
			classes01 = [[],[]]
			classes01[0].append( classes_dup.pop( self.rng.randint(0,len(classes_dup)-1) ))
			classes01[1].append( classes_dup.pop( self.rng.randint(0,len(classes_dup)-1) ))
			while(len(classes_dup) > 0):
				classes01[ self.rng.randint(0,1) ].append( classes_dup.pop() )
			
			tr_y_dup = [ 0 if sample in classes01[0] else 1 for sample in self.Tr_y]
			te_y_dup = [ 0 if sample in classes01[0] else 1 for sample in self.Te_y]


			model.fit( Tr_x, tr_y_dup, Te_x, te_y_dup)
			models_tmp.append( model.getPopulationsOverTime() )

			sot = model.getSizeOverTime()
			gt = model.getGenerationTimes()
			for i in range(self.max_generation):
				self.sizeOverTime[i] += sot[i]
				self.generationTimes[i] += gt[i]

			#print("%.4f --- %50s -- %s" % (self.modelos[-1][-1][0].getAccuracy(Tr_x, tr_y_dup), str(self.modelos[-1][-1][0]), str(classes01)  ))

		# Calcular as metricas usando o melhor modelo de cada gen em cada run como features:
		# self.modelos[run][geracao][populacao], populacao ordenada do melhor para o pior

		# models_tmp -> [run][gen][id]
		self.modelos = []
		for gen in range(self.max_generation):
			self.modelos.append([])
			for id_ in range(self.population_size):
				self.modelos[-1].append([models_tmp[r][gen][id_] for r in range(n_runs)])

		self.final_models = self.modelos[-1][0]


		self.trainingAccuracyOverTime = []
		self.testAccuracyOverTime = []
		self.fitnessOverTime = []
		self.trainingWaFOverTime = []
		self.testWaFOverTime = []
		self.trainingKappaOverTime = []
		self.testKappaOverTime = []
		self.trainingMSEOverTime = [0]*self.max_generation
		self.testMSEOverTime = [0]*self.max_generation

		for g in range(self.max_generation):
			tmp_models = []
			for r in range(n_runs):
				tmp_models.append(self.modelos[r][g][0])

			df_tr = pd.DataFrame()
			df_te = pd.DataFrame()
			for m in range(n_runs):
				df_tr["#%d"%m] = tmp_models[m].convert(Tr_x)
				df_te["#%d"%m] = tmp_models[m].convert(Te_x)
		
			if False:
				rf = RandomForestClassifier(random_state=42, max_depth=5)

			
				scaler = MinMaxScaler()
				for col in df_tr.columns:
					scaler = MinMaxScaler()
					scaler.fit(pd.DataFrame(df_tr[col]))
					df_tr[col] = pd.DataFrame(scaler.transform(pd.DataFrame(df_tr[col])),columns=[col])
					df_te[col] = pd.DataFrame(scaler.transform(pd.DataFrame(df_te[col])),columns=[col]).clip(lower=-1, upper=1)

				rf.fit(df_tr, Tr_y)


				tr_pred = rf.predict(df_tr)
				te_pred = rf.predict(df_te)
			

				self.trainingAccuracyOverTime.append( accuracy_score(Tr_y, tr_pred) )
				self.testAccuracyOverTime.append( accuracy_score(Te_y, te_pred) )
				self.trainingWaFOverTime.append( f1_score(Tr_y, tr_pred, average="weighted") )
				self.testWaFOverTime.append( f1_score(Te_y, te_pred, average="weighted") )
				self.trainingKappaOverTime.append( cohen_kappa_score(Tr_y, tr_pred) )
				self.testKappaOverTime.append( cohen_kappa_score(Te_y, te_pred) )
			else:
				self.trainingAccuracyOverTime.append(0 )
				self.testAccuracyOverTime.append( 0 )
				self.trainingWaFOverTime.append( 0 )
				self.testWaFOverTime.append( 0 )
				self.trainingKappaOverTime.append( 0 )
				self.testKappaOverTime.append( 0 )



		#print("Training: %.4f" % accuracy_score(Tr_y, rf.predict(df_tr)))
		#print("Test:     %.4f" % accuracy_score(Te_y, rf.predict(df_te)))

		# IM-3
		# As RF-base conseguem uma mediana de 95.87 com stdev 2.04
		# rbd + Accuracy (20gen) = 92.78±2.64
		# rbd + Pearson (20gen) = 94.85±3.15
		# rbd + Pearson (100gen)= 94.85±3.57



