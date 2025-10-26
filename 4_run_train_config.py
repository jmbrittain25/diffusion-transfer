import argparse
import json

from dojo import Experiment


def main(args):
    # TODO - open config file, load as json

    # Create and run experiment from json
    experiment = Experiment.from_json(json_str)
    experiment(output_dir)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transfer Model using Orbital Trajectories")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    args = parser.parse_args()
    main(args)