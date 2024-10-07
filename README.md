
** Data Splitting **
Example command: 
> python dataloader.py --language Thai --dpi 200 --style normal --training-ratio 0.8 --testing-ratio 0.1 --output-dir ./splits

** Model Training **
Example command: 
> python training.py --train-file ./splits/train.txt --valid-file ./splits/valid.txt --batch-size 32 --epochs 5 --lr 0.001 --output-model ./ocr_model.pth --device cuda

** Model Evaluation **
Example command:
> python evaluate.py --model_path ./ocr_model.pth --test_data_dir ./splits/test.txt --train_data_dir ./splits/train.txt --batch_size 32
