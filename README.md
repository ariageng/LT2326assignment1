
## **Data Splitting**
dataloader.py

Example command: 
```python dataloader.py --language Thai --dpi 200 --style normal --training-ratio 0.8 --testing-ratio 0.1 --output-dir ./splits```



> for special cases (e.g. training on "Thai normal text, 400dpi" and testing on	"Thai normal text, 200dpi"), create two splits:

```python dataloader.py --language Thai --dpi 400 --style normal --training-ratio 0.8 --testing-ratio 0.1 --output-dir ./trainsplits```

and ```python dataloader.py --language Thai --dpi 200 --style normal --training-ratio 0.8 --testing-ratio 0.1 --output-dir ./testsplits```

then train with the trainsplits ```python training.py --train-file ./trainsplits/train.txt --valid-file ./trainsplits/valid.txt --batch-size 32 --epochs 5 --lr 0.001 --output-model ./ocr_model.pth --device cuda```

but evaluate using the test in testsplits ```python evaluate.py --model_path ./ocr_model.pth --test_data_dir ./testsplits/t
est.txt --train_data_dir ./trainsplits/train.txt --batch_size 32```

## **Model Training**
training.py

Example command: 
```python training.py --train-file ./splits/train.txt --valid-file ./splits/valid.txt --batch-size 32 --epochs 5 --lr 0.001 --output-model ./ocr_model.pth --device cuda```

## **Model Evaluation**
evaluate.py

Example command:
```python evaluate.py --model_path ./ocr_model.pth --test_data_dir ./splits/test.txt --train_data_dir ./splits/train.txt --batch_size 32```

## Experiment results:
Training data	vs. Testing data (batch-size 32 --epochs 5 --lr 0.001)
#### Thai normal text, 200dpi	Thai normal text, 200dpi
Accuracy score: 0.935243553008596

#### Thai normal text, 400dpi	Thai normal text, 200dpi (yes, different resolution, figure out the logistics of this)
Accuracy score: 0.8813753581661891

#### Thai normal text, 400 dpi	Thai bold text, 400dpi
Accuracy score: 0.8054116292458261

#### Thai bold text	Thai normal text
Accuracy score: 0.9008317338451696

#### All Thai styles	All Thai styles
Accuracy score: 0.9296273827389124

#### Thai and English normal text jointly	Thai and English normal text jointly.
Accuracy score: 0.9467005076142132

#### All Thai and English styles jointly.	All Thai and English styles jointly.
Accuracy score: 0.710299288576335
