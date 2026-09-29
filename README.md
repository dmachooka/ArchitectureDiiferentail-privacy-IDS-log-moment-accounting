# ArchitectureDiiferentail-privacy-IDS-log-moment-accounting
This project implements a rigorous and reproducible Private Aggregation of Teacher Ensembles (PATE) privacy accounting mechanism for IoT intrusion detection using a tabular dataset. The primary goal is to evaluate the privacy-utility trade-off across various teacher and student model configurations.
2.2. Data Setup
Dataset: The experiment utilizes the 4 datasets mqtt,RT_IoT,IOTBotnet datasets. T All subsequent data preparation and preprocessing, including encoding, feature selection, missing value handling, and standardization, are performed by dedicated functions within this notebook, as detailed in Section 4.4.

Steps to obtain and set up the dataset:

Download the csv dataset. (Source: Please specify the original source or provide a download link if publicly available, e.g., a UCI repository link if applicable).
Upload the .csv file to your computer.
Ensure the DATA_PATH variable in the configuration section points to the correct location in your Google Drive. The default path is /content/drive/MyDrive/RT_IOT2022.csv. You will need to execute the cell from import drive; drive.mount('/content/drive') to mount your  Drive.
3. Experiment Configuration 
All critical parameters are defined at the beginning of the notebook to facilitate easy modification and replication. Key parameters include:

SEED: For reproducibility across all random operations.
DATA_PATH: Path to the dataset.
TARGET_COLUMN: The target feature for classification (Attack_type).
SELECTED_CLASSES: Specific classes of interest for the intrusion detection task.
NUM_TEACHERS, TEACHER_ARCHITECTURES, STUDENT_ARCHITECTURE: PATE model architecture definitions.
BATCH_SIZE, TEACHER_EPOCHS, STUDENT_EPOCHS, LEARNING_RATE, PATIENCE: Training hyperparameters.
TEACHER_VALID_FRACTION, STUDENT_VALID_FRACTION: Validation split ratios.
TEACHER_POOL_FRACTION, STUDENT_PRIVATE_FRACTION, PUBLIC_FRACTION, FINAL_TEST_FRACTION: Dataset partitioning ratios (sum to 1.0).
LAPLACE_INVERSE_SCALE: Parameter for Laplace noise in PATE aggregation.
DELTA, MAX_MOMENT: Privacy accounting parameters.
USE_OVERSAMPLING: Boolean to enable/disable RandomOverSampler for teacher training partitions.
RUN_FULL_EXPERIMENT: Flag to control execution of the full PATE experiment vs. smoke tests.
OUTPUT_DIR: Directory for saving results.
4. Experiment Steps (Detailed Procedure)
4.1. Reproducibility 
A set_seed function fixes random seeds (42) for random, numpy, and torch to ensure reproducible results across multiple runs.

4.2. Device Setup 
The DEVICE variable is set to cuda if a GPU is available, otherwise cpu. This optimizes training performance on Colab's infrastructure.

4.3. Data Loading 
The load_dataset function reads the .csv files, filters it to retain only the SELECTED_CLASSES, and prints the original and filtered dataset shapes and class distribution. This step ensures that the dataset matches the experimental scope.

4.4. Data Preprocessing and Splitting 75:15:10
The preprocess_and_split function performs critical data preparation:

Target Encoding: Converts categorical Attack_type labels into numerical format using LabelEncoder.
Feature Selection: Removes non-numeric columns (proto, service) as they are not suitable for the current neural network models.
Missing Value Handling: Converts to numeric and removes rows with NaN values.
Dataset Partitioning: Divides the data into four disjoint sets with controlled leakage:
X_teacher, y_teacher: Teacher training pool (75% of total data).
X_private, y_private: Student private labeled data (5%).
X_public, y_public: Public unlabeled data for teacher querying (5%).
X_final_test, y_final_test: Independent final test set (15%).
Standardization: StandardScaler is fitted only on the teacher pool (X_teacher) and then applied to all partitions. This prevents data leakage from test/public sets into the training process.
4.5. Custom Dataset Class 
A TabularDataset class, inheriting from torch.utils.data.Dataset, is defined to handle numerical features and labels, converting them into PyTorch tensors. This custom dataset facilitates batching and model training with DataLoader.

4.6. Teacher Data Loaders 
The create_teacher_loaders function partitions the X_teacher, y_teacher pool into NUM_TEACHERS disjoint subsets. Each teacher receives a unique training and validation split. If USE_OVERSAMPLING is true, RandomOverSampler is applied only to the training partition of each teacher to mitigate class imbalance without leaking synthetic data to other partitions or validation sets.

4.7. Model Definitions
This section defines the neural network architectures used for teachers and the student:

TeacherLSTM: Long Short-Term Memory network for sequential data processing.
TeacherGRU: Gated Recurrent Unit network, an alternative to LSTM.
TeacherCNN: Convolutional Neural Network for feature extraction from tabular data.
TeacherMLP: Multilayer Perceptron, a standard feed-forward network.
StudentMLP: Multilayer Perceptron used as the student model. All models output log-softmax probabilities for classification.
4.8. Model Factory 
The build_teacher function acts as a factory to instantiate teacher models based on the specified architecture string, input_size, and num_classes.

4.9. Training Function 
The train_model function encapsulates the training loop for both teachers and the student. It uses nn.NLLLoss as the criterion, optim.Adam as the optimizer, and implements early stopping based on validation loss to prevent overfitting. The best model state (lowest validation loss) is restored before the function returns.

4.10. Prediction Function 
The predict_labels function takes a trained model and a DataLoader, then generates integer class predictions by taking the argmax of the model's log-probabilities.

4.11. Vote Histogram Construction
teacher_predictions_to_counts converts the raw predictions from all teachers on the public dataset into a vote histogram, where each row represents a public query and columns represent class vote counts.

4.12. Laplace Report-NoisyMax Aggregation 
aggregate_teacher_votes_laplace implements the PATE aggregation mechanism. It adds Laplace noise (controlled by laplace_inverse_scale) to the teacher vote counts. The class with the highest noisy vote count is then released as the pseudo-label for the student.

4.13. Stable Log-Sum-Exp 
logaddexp_scalar is a utility function for numerically stable computation of log(exp(a) + exp(b)), crucial for privacy accounting calculations.

4.14. PATE Privacy Accounting (Log Moments) 
This section contains functions for computing log moments, which are fundamental to RDP (Rényi Differential Privacy) based accounting:

logmgf_exact: Computes an upper bound for a single query's privacy-loss log moment.
compute_q_noisy_max: Calculates q, the probability of incorrect consensus, based on vote margins and the Laplace noise scale.
data_dependent_log_moment: Computes the query-specific data-dependent privacy-loss log moment, leveraging the observed vote counts (q).
data_independent_log_moment: Computes the worst-case per-query log moment, independent of actual vote counts.
4.15. Complete PATE Privacy Accounting)
calculate_pate_privacy orchestrates the privacy accounting process. It takes teacher predictions or vote counts, applies the log moment calculations, and composes them over all released queries to determine the cumulative data-dependent and data-independent epsilon. It also returns diagnostic information such as q_values and vote_margins.

4.16. PATE Validation Tests
The run_accounting_tests function includes a suite of assertions to validate the correctness of the PATE privacy accounting implementation, covering vote histogram construction, permutation invariance, consensus impact on epsilon, and cumulative privacy loss.

4.17. Student Training Data Preparation 
prepare_student_data combines the private labeled data (from X_private, y_private) and the PATE pseudo-labeled public data (X_public, pseudo_labels) to form the student's training set. A validation split is created from the private labeled data.

4.18. Final Model Evaluation 
evaluate_model assesses the final student model's performance on the completely held-out X_final_test set. It reports accuracy, precision, recall, F1-score (macro averages), and the confusion matrix.

4.19. Complete Experiment Execution 
The run_full_pate_experiment function integrates all the above steps: data loading, preprocessing, teacher training, generating teacher predictions, PATE aggregation, privacy accounting, student training, and final evaluation. It saves comprehensive summary results, query-level diagnostics, and student predictions to CSV files in the OUTPUT_DIR.

4.20. Sensitivity Analysis 
The notebook includes sections for sensitivity analysis to study the impact of architectural configurations (e.g., varying number and types of teachers and student architectures) and the number of teachers on the privacy-utility tradeoff. This involves iterating through predefined configurations, updating global parameters (NUM_TEACHERS, TEACHER_ARCHITECTURES, STUDENT_ARCHITECTURE), and re-running run_full_pate_experiment.

4.21. Visualization and Correlation 
Various plots are generated to visualize key metrics:

Epsilon candidates over moment orders.
Relationship between mean vote margin and student accuracy.
Comparison of data-dependent and data-independent epsilon across different numbers of teachers.
Mean vote margin across different numbers of teachers.
Student accuracy across different teacher counts.
Correlation between the number of teachers and data-dependent epsilon. Correlation coefficients are also calculated to quantify relationships between variables.
5. How to Run the Experiment
Open in jupyter: Upload the .ipynb file to  open it directly.

Place Dataset: Ensure thecsv file is in the path specified by DATA_PATH (default: /content/drive/Drive/.csv).
Execute Cells Sequentially: Run all cells in the notebook from top to bottom. The notebook is structured to execute the accounting tests first, and then the full experiment if RUN_FULL_EXPERIMENT is set to True.
Monitor Progress: Observe the console output for training progress, privacy accounting results, and final evaluation metrics.
Review Results: After execution, check the pate_results directory (created automatically) for pate_privacy_summary.csv, pate_query_diagnostics.csv, and student_final_predictions.csv.
6. Expected Output and Interpretation
Upon successful execution, you should observe:

Console Output: Detailed training logs for each teacher and the student, privacy accounting results (data-dependent and data-independent epsilon), and final student model evaluation metrics (accuracy, F1-macro, confusion matrix).
Saved CSV Files (in ./pate_results/):
pate_privacy_summary.csv: A single row summary of the entire experiment's configuration and key results.
pate_query_diagnostics.csv: Per-query details including q values, vote margins, released labels, and true public labels.
student_final_predictions.csv: True vs. predicted labels for the final test set.
Plots: Various visualizations illustrating privacy loss composition, the relationship between vote margin and accuracy, and trends related to the number of teachers.
Interpretation of Results:
Epsilon Values: Lower epsilon values indicate stronger privacy guarantees. Data-dependent epsilon (epsilon_DD) typically provides a tighter bound than data-independent epsilon (epsilon_DI), reflecting the actual consensus among teachers.
Student Accuracy: Measures the utility of the privately trained student model.
Mean Vote Margin: A higher mean vote margin indicates stronger teacher consensus, generally leading to better privacy guarantees.
q-values: Represent the probability of an incorrect label being selected by Report-NoisyMax. Lower q values contribute to better privacy.
7. Rigor and Bias Mitigation
To ensure rigor and mitigate bias, several practices are followed:

Reproducibility: Strict seeding (SEED) for all random operations (data splitting, model initialization, noise addition) ensures that the experiment can be exactly replicated.
Controlled Data Splitting: The preprocess_and_split function explicitly controls data leakage by performing scaling after splitting and fitting the scaler only on the teacher pool. The partitioning ensures disjoint sets for teacher training, student private training, public querying, and final testing.
Disjoint Teacher Datasets: Each teacher receives a unique, disjoint subset of the teacher pool, preventing training data overlap among teachers.
Conditional Oversampling: RandomOverSampler is applied only to the training portion of each teacher's dataset, and only if USE_OVERSAMPLING is enabled, to prevent artificial inflation of validation/test metrics or leakage across teachers.
Independent Final Test Set: A completely independent test set (FINAL_TEST_FRACTION) is used for unbiased evaluation of the student model's generalization ability, ensuring it has not been seen during any part of the training or privacy accounting process.
Established Privacy Accounting: The legacy Laplace Report NoisyMax moments accountant is a well-established method in differential privacy research, providing a formally verifiable privacy guarantee.
Transparency: All configuration parameters, code, and result saving mechanisms are clearly defined and documented, promoting transparency and auditability.
8. Conclusion
This notebook provides a robust framework for understanding and replicating PATE privacy accounting in the context of IoT intrusion detection. By adhering to the outlined steps and considering the detailed explanations, researchers can reproduce the results, explore variations, and build upon this foundation with confidence in its methodological soundness.
