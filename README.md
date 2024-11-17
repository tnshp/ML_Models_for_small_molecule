# Molecular Dynamics Prediction Using SchNet and sGDML
This project aims to evaluating two state-of-the-art machine learning models, SchNet and sGDML, for their ability to predict molecular forces and potential energy surfaces. It aims to replicate and extend the findings from the study by Vassilev-Galindo, Valentin, et al., titled "Challenges for machine learning force fields in reproducing potential energy surfaces of flexible molecules" (The Journal of Chemical Physics, 2021). The focus is on using the same dataset and preparing the models for using them to make predictions on various others organic molecules. The report on the project work explores the implementation challenges, computational requirements, and model performance, highlighting key observations and insights related to molecular dynamics predictions.


## SchNet

### Installation
To get started with SchNet, first install **SchNetPack**. You can install it by running the following commands:

```bash
pip install schnetpack
```

Make sure to install all the dependencies required for SchNet. You can refer to the official [SchNetPack documentation](https://www.schnetpack.org/) for detailed installation instructions.

### Dataset
You can download the dataset from the following link: [Dataset Link](https://pubs.aip.org/jcp/article-supplement/313847/zip/094119_1_supplements/). I have provided 1 file for reference here.

### Workflow
1. **Convert XYZ Dataset Files to NPZ**: Before you start training, you need to convert the XYZ files to NPZ files. This can be done by running the script `xyz_npz.py`.
   ```bash
   python xyz_npz.py
   ```

2. **Create the Database File**: After converting the dataset, create a `.db` file, which will act as the database file for feeding into the model. This can be done using the script `create_db.py`.
   ```bash
   python create_db.py
   ```

3. **Training the Model**: After preparing the data, you can now train the model.

   - If you have a GPU, run the following command to train the model for predicting forces:
     ```bash
     python main_gpu.py
     ```

   - If you don't have a GPU, you can run the following command:
     ```bash
     python main.py
     ```

   - If you want to train a model for predicting energies, use this command:
     ```bash
     python main_energy.py
     ```

4. **Tuning Hyperparameters**: You can tune various hyperparameters like the number of basis atoms, number of interactions, cutoff radius, etc., to see how different configurations affect the model performance.

5. **Model Saved**: Once the training is complete, the model will be saved in a `.pth` file.

6. **Making Predictions**: After training, you can use the saved model to make predictions. Run the following command to get the Mean Squared Error (MSE) and Mean Absolute Error (MAE) using `test.py`:
   ```bash
   python test.py
   ```

## sGDML

### Installation
To get started with sGDML, make sure **sGDML** is installed with all the required dependencies. You can find the installation instructions on the official [sGDML GitHub page](https://github.com/stefanch/sGDML).

### Workflow
1. **Load Dataset**: First, you need to load the dataset from an XYZ file into an NPZ file. This can be done using the provided script `xyz_to_npz.py`.
   ```bash
   python xyz_to_npz.py
   ```

2. **Training the Model**: Once you have the dataset in NPZ format, you can start training the model. Run the following command to start training:
   ```bash
   python train.py
   ```
   This will create a `.pth` file containing the trained model.

3. **Prediction**: To use the trained model for prediction, run the following command to create an output file `output.txt` with the predictions:
   ```bash
   python predict.py
   ```

4. **Test Model**: Alternatively, you can use `test.py` to directly compute the MSE and MAE for the trained model:
   ```bash
   python test.py
   ```

### Notes
Make sure that the paths for the input files and the model are correctly set in the Python scripts before running them.
