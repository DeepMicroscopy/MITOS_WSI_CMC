import pickle
import sys
from SlideRunner.dataAccess.database import Database
f = pickle.load(open(sys.argv[1],'rb'))

if True:#len(sys.argv)>2:
    slidelist_test_1 = ['14','18','3','22','10','15','21']
    slidelist_test_2 = ['1','20','17','5','2','11','16']
    slidelist_test_3 = ['13','7','19','8','6','9', '12']
#    slidelist_test_1 = ['18','3','22','10','15','21','12']
#    slidelist_test_2 = ['1','20','17','5','2','11','16']
#    slidelist_test_3 = ['14','13','7','19','8','6','9']
    
    if (sys.argv[2] == '1'):
        slidelist_test = slidelist_test_1
    elif (sys.argv[2] == '2'):
        slidelist_test = slidelist_test_2
    elif (sys.argv[2] == '3'):
        slidelist_test = slidelist_test_3

DB = Database()
DB.open('databases/MITOS_WSI_CMC_ODAEL_TR.sqlite')
DB.execute('SELECT uid, filename FROM Slides')
slides = DB.fetchall()
slidelist_train = [y[1] for y in slides if str(y[0]) not in slidelist_test]
slidelist_test = [y[1] for y in slides if str(y[0]) in slidelist_test]
print('Training:',slidelist_train)
print('Test:', slidelist_test)

test = {x:f[x] for x in slidelist_test}
train = {x:f[x] for x in slidelist_train}

pickle.dump(train, open('trainval_'+sys.argv[1],'wb'))
pickle.dump(test, open('test_'+sys.argv[1],'wb'))
