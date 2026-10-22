import pandas as pd 
from config import conf
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import utils
from models.ensemble import kotsiantis
"""
    Tests a deep learning module

    Params:
        - net (nn.Module): the network to test
        - loader (DataLoader): the data loader of the testing instances
        - performances (pd.DataFrame): the dataframe where to report the
            performances

    return the performances dataframe (pd.DataFrame)
"""
def test_dl(net, loader, performances):
	y_pred = list()
	y_true = list()
    
	for x_batch, y_batch in loader:
		if conf.GPU == 1 and conf.DEEP_LEARN == True:
			x_batch = x_batch.cuda()
			y_batch = y_batch.cuda()
        
		net.eval()
		if conf.MODEL.__name__ == 'HAN':
			net._init_h_state(last_batch_size=len(x_batch), init=-1)
		y_batch_hat = net(x_batch)
		y_batch_hat.squeeze_(1)
		y_true += list(y_batch.data.numpy().flatten())
		y_pred += list(y_batch_hat.data.numpy().flatten())
	aucpr = average_precision_score(y_true, y_pred)
	aucroc = roc_auc_score(y_true, y_pred)
	y_true = '[' + ' '.join([str(e) for e in y_true]) + ']'
	y_pred = '[' + ' '.join([str(e) for e in y_pred]) + ']'
    
	return pd.concat([performances, pd.DataFrame({'y_true':y_true,\
        'y_pred':y_pred, 'aucpr':aucpr, 'aucroc':aucroc}, index=[0])])

"""
     Tests a machine learning model

    Params:
        - net : the model to test
        - X (np.ndarray): the data corresponding to the test instances
        - y (np.array): the ground truth labels
        - performances (pd.DataFrame): the dataframe where to report the
            performances

    return the performances dataframe (pd.DataFrame)
"""


def test_ml(net, dataset, performances):
	X, y = dataset

	y_pred = net.predict(X,y) \
		if conf.MODEL.__name__ == 'MajorityClassBaseline'\
			else net.predict(X)
	
	
	average_precision = average_precision_score(y, y_pred)
	precision, recall, thresholds = precision_recall_curve(y, y_pred)
	auc_precision_recall = auc(recall, precision)
	y_true = '[' + ' '.join([str(e) for e in y]) + ']'
	y_pred = '[' + ' '.join([str(e) for e in y_pred]) + ']'
    
	explanations = integrate_explainability(model, X)
	return pd.concat([performances,\
        pd.DataFrame({'y_true':y_true, 'y_pred':y_pred, 'prauc':average_precision, 'prc':auc_precision_recall, 'Explainability':explanations}, index=[0])])

#def evaluate_model(model, dataset):
   # X, y = dataset
   # predictions = model.predict(X)
   ## accuracy = np.mean(predictions == y)
    #print(f'Accuracy: {accuracy:.2f}')

    # Generate explainability
    #explanations = integrate_explainability(model, X)
    #print(f'Explainability:', explanations)
	
