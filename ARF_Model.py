from skmultiflow.data import FileStream
from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.evaluation import EvaluatePrequential
# 1. Create a stream
stream = FileStream("All.csv")
stream.prepare_for_use()

# 2. Instantiate the HoeffdingTreeClassifier
ada =AdaptiveRandomForest()
## 3. Setup the evaluator
evaluator = EvaluatePrequential(pretrain_size=1000,
                                max_samples=110600,
                                output_file='results3.csv')
## 4. Run evaluation
evaluator.evaluate(stream=stream, model=ada)
