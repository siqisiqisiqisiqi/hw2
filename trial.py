import os
import pandas as pd
test_feat_path = 'test_data'
test_text_path = os.path.join(test_feat_path, 'testing_label.json')
print(test_text_path)
data = pd.read_json(test_text_path)
print(data)