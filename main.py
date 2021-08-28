import argparse
import os
from datetime import datetime
from data import get_dataloaders
from model import initialize_model
from train import train_model


def main(args):
    # Determine the number of classes
    classes = [
        lbl
        for lbl in os.listdir(args.data)
        if os.path.isdir(os.path.join(args.data, lbl))
    ]
    args.num_classes = len(classes)

    # Create output directory
    cur_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, cur_datetime)
    os.makedirs(args.output_dir, exist_ok=True)

    # Get model
    model, args.input_size = initialize_model(
        args.model_name, args.num_classes, args.finetune
    )

    # Get dataset
    dataloaders = {}
    dataloaders["train"], dataloaders["val"] = get_dataloaders(args)

    # Train model
    model = train_model(
        args, model, dataloaders, is_inception=args.model_name == "inception"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trash classification")
    parser.add_argument("--data", type=str, default="trash-data/dataset-resized")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--finetune", action="store_false")
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet",
        choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"],
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    main(args)
