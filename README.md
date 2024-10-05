Example command for data splitting: python dataloader.py --language Thai --dpi 200 --style normal --training-ratio 0.8 --testing-ratio 0.1 --output-dir ./splits

Example command for model training: python training.py --train-file ./splits/train.txt --valid-file ./splits/valid.txt --batch-size 32 --epochs 5 --lr 0.001 --output-model ./ocr_model.pth --device cuda
