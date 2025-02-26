import os
from gluonts.dataset.split import split
from utils.data_loader import load_eeg_data, prepare_dataset
from models.moirai_classifier import SleepStagePredictor
from config.config import ModelConfig, DataConfig
import matplotlib.pyplot as plt
from uni2ts.eval_util.plot import plot_single

def main():
    # Load data
    df = load_eeg_data(DataConfig.EDF_PATH, DataConfig.LABEL_PATH)
    dataset = prepare_dataset(df)
    
    # Split dataset
    train_length = int(DataConfig.TRAIN_SPLIT * len(dataset))
    train, test = split(dataset, offset=-train_length)
    
    # Initialize model
    model = SleepStagePredictor(ModelConfig)
    predictor = model.create_predictor()
    
    # Make predictions
    forecasts = predictor.predict(test.input)
    
    # Plot results
    input_it = iter(test.input)
    label_it = iter(test.label)
    forecast_it = iter(forecasts)

    inp = next(input_it)
    label = next(label_it)
    forecast = next(forecast_it)

    plot_single(
        inp, 
        label, 
        forecast, 
        context_length=ModelConfig.CONTEXT_LENGTH,
        name="Sleep Stage Prediction",
        show_label=True,
    )
    plt.show()

if __name__ == "__main__":
    main()
