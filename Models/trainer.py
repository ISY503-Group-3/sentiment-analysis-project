#!/usr/bin/env python3
"""
Background trainer CLI for running TensorFlow-heavy work outside the Streamlit process.

Usage examples:
  python Models/trainer.py --csv data/input_demo.csv --epochs 3 --batch-size 32

For macOS Apple Silicon it's recommended to run this inside a conda env with a
supported Python (3.10 or 3.11) and `tensorflow-macos` + `tensorflow-metal` installed.
Example wrapper script: `scripts/run_trainer_conda.sh`.
"""
import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description='Background trainer CLI — runs tournament and final training.')
    parser.add_argument('--csv', help='Path to input CSV (cleaned) to train on', default='data/input_demo.csv')
    parser.add_argument('--epochs', type=int, default=3, help='Tournament epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--final-epochs', type=int, default=20, help='Final training epochs')
    parser.add_argument('--out-dir', default='artifacts', help='Directory to save artifacts')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Defer heavy imports until we're in this separate process
    try:
        from Models import model_architect_full as ma
    except Exception as e:
        print(f"FATAL: Could not import Models.model_architect_full: {e}", file=sys.stderr)
        sys.exit(2)

    # Set env guards before importing TF
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('KMP_BLOCKTIME', '1')

    try:
        print('Attempting to import TensorFlow (this may fail if environment is incompatible)...')
        ma.ensure_tf()
    except Exception as e:
        print(f"TensorFlow import failed: {e}", file=sys.stderr)
        sys.exit(3)

    try:
        print(f'Loading CSV: {args.csv}')
        df = ma.load_data(args.csv)
    except Exception as e:
        print(f"Could not load CSV '{args.csv}': {e}", file=sys.stderr)
        sys.exit(4)

    print('Preprocessing and tokenizing...')
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = ma.tokenize_split_and_save(df, save_artifacts=True,
                                                                                               tokenizer_path=os.path.join(args.out_dir, 'tokenizer.pkl'),
                                                                                               label_map_path=os.path.join(args.out_dir, 'label_map.json'))
    except Exception as e:
        print(f"Preprocessing failed: {e}", file=sys.stderr)
        sys.exit(5)

    print('Running tournament...')
    try:
        res_df = ma.run_tournament(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)
        print('Tournament results:')
        print(res_df[['Model Name', 'Val Accuracy', 'Time (s)']].to_string(index=False))
    except Exception as e:
        print(f"Tournament failed: {e}", file=sys.stderr)
        sys.exit(6)

    print('Selecting final model...')
    try:
        final_model = ma.select_final_model(res_df)
        if final_model is None:
            print('No final model selected. Exiting.')
            sys.exit(0)
    except Exception as e:
        print(f"Model selection failed: {e}", file=sys.stderr)
        sys.exit(7)

    print('Starting final training... (this may take a while)')
    try:
        out = ma.final_train_and_evaluate(final_model, X_train, y_train, X_val, y_val, X_test, y_test,
                                          epochs=args.final_epochs, batch_size=args.batch_size)
        print('Final training complete. Evaluation:')
        evald = out['evaluation']
        # Print ROC AUC and test accuracy summary similar to notebook
        print(f"ROC AUC: {evald.get('roc_auc')}")
        test_acc = None
        try:
            test_acc = evald.get('report', {}).get('accuracy')
        except Exception:
            test_acc = None
        if test_acc is not None:
            print(f"FINAL TEST ACCURACY: {float(test_acc)*100:.2f}%")
            if float(test_acc) > 0.90:
                print('\n🌟 HIGH DISTINCTION TARGET MET (> 90%)')
            else:
                print('\n⚠️ WARNING: < 90%. Consider retraining with more epochs.')
        # Save artifacts
        ma.save_artifacts(out['model'], tokenizer, {0: 'negative', 1: 'positive'},
                          model_path=os.path.join(args.out_dir, 'final_sentiment_model.keras'),
                          tokenizer_path=os.path.join(args.out_dir, 'tokenizer.pkl'),
                          label_map_path=os.path.join(args.out_dir, 'label_map.json'))
        print(f"Artifacts saved to: {args.out_dir}")
    except Exception as e:
        print(f"Final training or saving failed: {e}", file=sys.stderr)
        sys.exit(8)


if __name__ == '__main__':
    main()
