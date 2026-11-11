import os
import sys
import tempfile
import unittest
from unittest import mock

import cli
from progress import format_duration, format_number, timing


class ProgressTest(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(format_duration(5.4), "5s")
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(format_duration(3725), "1h 02m")

    def test_format_number(self):
        self.assertEqual(format_number(0.048424858277279687), "0.04842")
        self.assertEqual(format_number(1.8571100477077985e-30), "1.857e-30")
        self.assertEqual(format_number(None), "-")

    def test_time_left_assumes_the_same_pace(self):
        with mock.patch("progress.time.time", return_value=100.0):
            self.assertEqual(timing(start_time=40.0, done=3, total=10), "1m 00s elapsed, 2m 20s left")  # 20 s per item
            self.assertEqual(timing(start_time=100.0, done=0, total=10), "0s elapsed, 0s left")


class CliTest(unittest.TestCase):
    def test_project_path_doesnt_depend_on_the_current_directory(self):
        expected = os.path.join(os.path.dirname(os.path.abspath(cli.__file__)), "models", "x.joblib")

        with tempfile.TemporaryDirectory() as directory:
            current = os.getcwd()
            os.chdir(directory)
            try:
                self.assertEqual(cli.project_path("models", "x.joblib"), expected)
            finally:
                os.chdir(current)

    def test_training_args(self):
        with mock.patch.object(sys, "argv", ["train.py", "--seed", "3", "--episodes", "20"]):
            args = cli.training_args("test", render=True, episodes=500)
        self.assertEqual((args.seed, args.render, args.episodes), (3, False, 20))

        with mock.patch.object(sys, "argv", ["train.py"]):
            args = cli.training_args("test", epochs=10)
        self.assertEqual((args.seed, args.epochs), (None, 10))
        self.assertFalse(hasattr(args, "render"))

    def test_training_length_must_be_positive(self):
        with mock.patch.object(sys, "argv", ["train.py", "--steps", "0"]), mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):  # argparse prints the error and exits
                cli.training_args("test", steps=1000)

    def test_missing_model_explains_how_to_train_it(self):
        with self.assertRaises(SystemExit) as error:
            cli.load_trained_model(os.path.join("models", "missing.joblib"), "train-weather.py")

        self.assertIn("train it first with python train-weather.py", str(error.exception))


if __name__ == "__main__":
    unittest.main()
