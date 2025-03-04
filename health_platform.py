import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
import logging

class MultiModalHealthPlatform:
    """
    A platform for integrating and analyzing multiple types of healthcare data
    to create personalized recommendations for women's health.
    """

    def __init__(self, config=None):
        """
        Initialize the platform with configuration settings.

        Parameters:
        -----------
        config : dict
            Configuration parameters for the platform
        """
        # Default configuration
        self.config = {
            'embedding_dim': 128,
            'num_clusters': 5,
            'pca_components': 20,
            'random_state': 42,
            'n_init': 10
        }

        # Update with user configuration if provided
        if config:
            self.config.update(config)

        # Initialize components
        self.encoders = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.hierarchical_model = None
        self.gmm_model = None
        self.embedding_model = None

        # Setup logging
        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('MultiModalHealthPlatform')

    def preprocess_structured_data(self, data):
        """
        Preprocess structured clinical and demographic data.

        Parameters:
        -----------
        data : pandas.DataFrame
            Structured healthcare data including vitals, lab results, demographics, etc.

        Returns:
        --------
        pandas.DataFrame
            Preprocessed structured data
        """
        self.logger.info("Preprocessing structured data...")

        # Handle missing values
        data_processed = data.copy()

        # Impute missing numeric values with median
        numeric_cols = data_processed.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            data_processed[col].fillna(data_processed[col].median(), inplace=True)

        # Impute missing categorical values with mode
        categorical_cols = data_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            data_processed[col].fillna(data_processed[col].mode()[0], inplace=True)

        # One-hot encode categorical variables
        data_processed = pd.get_dummies(data_processed, columns=categorical_cols)

        # Scale numeric features
        data_processed[numeric_cols] = self.scaler.fit_transform(data_processed[numeric_cols])

        return data_processed

    def preprocess_unstructured_data(self, clinical_notes):
        """
        Process unstructured clinical notes using NLP techniques.

        Parameters:
        -----------
        clinical_notes : list
            List of clinical notes as text strings

        Returns:
        --------
        numpy.ndarray
            Document embeddings for clinical notes
        """
        self.logger.info("Preprocessing unstructured clinical notes...")

        # Download NLTK resources if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')

        # Tokenize and clean text
        stop_words = set(stopwords.words('english'))

        def preprocess_text(text):
            tokens = nltk.word_tokenize(text.lower())
            return [word for word in tokens if word.isalpha() and word not in stop_words]

        processed_notes = [preprocess_text(note) for note in clinical_notes]

        # Create document embeddings using Doc2Vec
        tagged_data = [TaggedDocument(words=note, tags=[str(i)])
                      for i, note in enumerate(processed_notes)]

        doc2vec_model = Doc2Vec(vector_size=self.config['embedding_dim'],
                               window=5,
                               min_count=2,
                               workers=4,
                               epochs=40)

        doc2vec_model.build_vocab(tagged_data)
        doc2vec_model.train(tagged_data,
                           total_examples=doc2vec_model.corpus_count,
                           epochs=doc2vec_model.epochs)

        # Generate document vectors
        doc_vectors = np.array([doc2vec_model.infer_vector(note) for note in processed_notes])

        # Save the model for future inference
        self.encoders['clinical_notes'] = doc2vec_model

        return doc_vectors

    def preprocess_wearable_data(self, wearable_data):
        """
        Process time-series data from wearable devices.

        Parameters:
        -----------
        wearable_data : dict
            Dictionary with patient IDs as keys and time series dataframes as values

        Returns:
        --------
        pandas.DataFrame
            Extracted features from wearable data
        """
        self.logger.info("Processing wearable device data...")

        # Extract features from time-series data
        features = []

        for patient_id, data in wearable_data.items():
            patient_features = {'patient_id': patient_id}

            # Calculate statistics for each measurement type
            for column in data.columns:
                if column != 'timestamp':
                    # Basic statistics
                    patient_features[f'{column}_mean'] = data[column].mean()
                    patient_features[f'{column}_std'] = data[column].std()
                    patient_features[f'{column}_min'] = data[column].min()
                    patient_features[f'{column}_max'] = data[column].max()

                    # Calculate daily averages and variations
                    if 'timestamp' in data.columns:
                        data['date'] = pd.to_datetime(data['timestamp']).dt.date
                        daily_avg = data.groupby('date')[column].mean()
                        patient_features[f'{column}_daily_variation'] = daily_avg.std()

            features.append(patient_features)

        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        features_df.set_index('patient_id', inplace=True)

        # Scale features
        numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
        features_df[numeric_cols] = StandardScaler().fit_transform(features_df[numeric_cols])

        return features_df

    def preprocess_survey_data(self, survey_data):
        """
        Process patient-reported outcomes and preferences.

        Parameters:
        -----------
        survey_data : pandas.DataFrame
            Survey responses including preferences and health goals

        Returns:
        --------
        pandas.DataFrame
            Processed survey data
        """
        self.logger.info("Processing survey and preference data...")

        survey_processed = survey_data.copy()

        # Handle Likert scales and categorical data
        likert_columns = [col for col in survey_processed.columns
                         if survey_processed[col].dtype == 'object' and
                         set(survey_processed[col].unique()).issubset(['Strongly Disagree', 'Disagree',
                                                                     'Neutral', 'Agree', 'Strongly Agree'])]

        likert_mapping = {
            'Strongly Disagree': 1,
            'Disagree': 2,
            'Neutral': 3,
            'Agree': 4,
            'Strongly Agree': 5
        }

        for col in likert_columns:
            survey_processed[col] = survey_processed[col].map(likert_mapping)

        # One-hot encode remaining categorical variables
        categorical_cols = survey_processed.select_dtypes(include=['object']).columns
        survey_processed = pd.get_dummies(survey_processed, columns=categorical_cols)

        return survey_processed

    def integrate_data_sources(self, structured_data, unstructured_embeddings,
                              wearable_features, survey_data, patient_ids):
        """
        Combine all data sources into a unified representation.

        Parameters:
        -----------
        structured_data : pandas.DataFrame
            Processed structured clinical data
        unstructured_embeddings : numpy.ndarray
            Document embeddings from clinical notes
        wearable_features : pandas.DataFrame
            Features extracted from wearable device data
        survey_data : pandas.DataFrame
            Processed survey and preference data
        patient_ids : list
            List of patient IDs to ensure alignment

        Returns:
        --------
        numpy.ndarray
            Integrated data representation
        """
        self.logger.info("Integrating multiple data sources...")

        # Ensure all data sources are aligned by patient ID
        integrated_data = []

        for i, patient_id in enumerate(patient_ids):
            # Get data for this patient from each source
            patient_structured = structured_data.loc[patient_id].values if patient_id in structured_data.index else np.zeros(structured_data.shape[1])
            patient_notes = unstructured_embeddings[i] if i < len(unstructured_embeddings) else np.zeros(self.config['embedding_dim'])
            patient_wearable = wearable_features.loc[patient_id].values if patient_id in wearable_features.index else np.zeros(wearable_features.shape[1])
            patient_survey = survey_data.loc[patient_id].values if patient_id in survey_data.index else np.zeros(survey_data.shape[1])

            # Combine all features for this patient
            patient_integrated = np.concatenate([
                patient_structured,
                patient_notes,
                patient_wearable,
                patient_survey
            ])

            integrated_data.append(patient_integrated)

        integrated_array = np.array(integrated_data)

        # Apply dimensionality reduction
        self.pca = PCA(n_components=self.config['pca_components'], random_state=self.config['random_state'])
        reduced_data = self.pca.fit_transform(integrated_array)

        self.logger.info(f"Data integration complete. Final dimensions: {reduced_data.shape}")

        return reduced_data

    def build_deep_embedding_model(self, input_shapes):
        """
        Build a deep learning model to create unified embeddings of heterogeneous data.

        Parameters:
        -----------
        input_shapes : dict
            Dictionary with input shapes for each data type

        Returns:
        --------
        tensorflow.keras.Model
            Trained embedding model
        """
        self.logger.info("Building deep embedding model for data integration...")

        # Create inputs for each data type
        structured_input = Input(shape=(input_shapes['structured'],), name='structured_input')
        notes_input = Input(shape=(input_shapes['notes'],), name='notes_input')
        wearable_input = Input(shape=(input_shapes['wearable'],), name='wearable_input')
        survey_input = Input(shape=(input_shapes['survey'],), name='survey_input')

        # Process each input type
        structured_features = Dense(64, activation='relu')(structured_input)
        structured_features = Dense(32, activation='relu')(structured_features)

        notes_features = Dense(64, activation='relu')(notes_input)
        notes_features = Dense(32, activation='relu')(notes_features)

        wearable_features = Dense(64, activation='relu')(wearable_input)
        wearable_features = Dense(32, activation='relu')(wearable_features)

        survey_features = Dense(64, activation='relu')(survey_input)
        survey_features = Dense(32, activation='relu')(survey_features)

        # Combine all features
        combined = Concatenate()([
            structured_features,
            notes_features,
            wearable_features,
            survey_features
        ])

        # Create unified embedding
        embedding = Dense(128, activation='relu')(combined)
        embedding = Dense(64, activation='relu')(embedding)
        embedding = Dense(self.config['embedding_dim'], name='embedding')(embedding)

        # Create model
        model = Model(
            inputs=[structured_input, notes_input, wearable_input, survey_input],
            outputs=embedding
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.logger.info(f"Embedding model built with output dimension: {self.config['embedding_dim']}")

        return model

    def apply_hierarchical_clustering(self, integrated_data):
        """
        Apply hierarchical clustering to the integrated data.

        Parameters:
        -----------
        integrated_data : numpy.ndarray
            Integrated and dimensionality-reduced data

        Returns:
        --------
        numpy.ndarray
            Cluster assignments
        """
        self.logger.info("Applying hierarchical clustering...")

        # Determine optimal number of clusters using silhouette score if not specified
        if self.config.get('auto_determine_clusters', False):
            max_score = -1
            optimal_clusters = self.config['num_clusters']

            for n_clusters in range(2, 11):
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clustering.fit_predict(integrated_data)
                score = silhouette_score(integrated_data, cluster_labels)

                if score > max_score:
                    max_score = score
                    optimal_clusters = n_clusters

            self.logger.info(f"Optimal number of clusters determined: {optimal_clusters}")
            self.config['num_clusters'] = optimal_clusters

        # Apply hierarchical clustering
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=self.config['num_clusters'],
            linkage='ward'
        )

        hierarchical_labels = self.hierarchical_model.fit_predict(integrated_data)

        return hierarchical_labels

    def apply_gmm_clustering(self, integrated_data):
        """
        Apply Gaussian Mixture Model clustering to the integrated data.

        Parameters:
        -----------
        integrated_data : numpy.ndarray
            Integrated and dimensionality-reduced data

        Returns:
        --------
        numpy.ndarray
            Cluster assignments and probabilities
        """
        self.logger.info("Applying Gaussian Mixture Model clustering...")

        # Initialize and fit GMM
        self.gmm_model = GaussianMixture(
            n_components=self.config['num_clusters'],
            random_state=self.config['random_state'],
            n_init=self.config['n_init']
        )

        self.gmm_model.fit(integrated_data)

        # Get cluster assignments and probabilities
        gmm_labels = self.gmm_model.predict(integrated_data)
        gmm_probs = self.gmm_model.predict_proba(integrated_data)

        return gmm_labels, gmm_probs

    def analyze_clusters(self, integrated_data, hierarchical_labels, gmm_labels,
                        gmm_probs, original_features, patient_ids):
        """
        Analyze the characteristics of each cluster.

        Parameters:
        -----------
        integrated_data : numpy.ndarray
            Integrated data
        hierarchical_labels : numpy.ndarray
            Cluster assignments from hierarchical clustering
        gmm_labels : numpy.ndarray
            Cluster assignments from GMM
        gmm_probs : numpy.ndarray
            Cluster probabilities from GMM
        original_features : pandas.DataFrame
            Original feature names for interpretation
        patient_ids : list
            List of patient IDs

        Returns:
        --------
        dict
            Cluster analysis results
        """
        self.logger.info("Analyzing cluster characteristics...")

        # Create DataFrame with patient IDs, cluster assignments, and feature values
        cluster_df = pd.DataFrame({
            'patient_id': patient_ids,
            'hierarchical_cluster': hierarchical_labels,
            'gmm_cluster': gmm_labels
        })

        # Add probabilities of belonging to each GMM cluster
        for i in range(self.config['num_clusters']):
            cluster_df[f'gmm_prob_cluster_{i}'] = gmm_probs[:, i]

        # Combine with original features for interpretation
        if isinstance(original_features, pd.DataFrame):
            cluster_df = pd.concat([cluster_df.set_index('patient_id'),
                                   original_features], axis=1).reset_index()

        # Analyze characteristics of each cluster
        cluster_analysis = {}

        # For each clustering method
        for method in ['hierarchical_cluster', 'gmm_cluster']:
            cluster_analysis[method] = {}

            # For each cluster
            for cluster_id in range(self.config['num_clusters']):
                cluster_members = cluster_df[cluster_df[method] == cluster_id]

                # Skip if no members in this cluster
                if len(cluster_members) == 0:
                    continue

                # Calculate basic statistics for this cluster
                cluster_analysis[method][cluster_id] = {
                    'size': len(cluster_members),
                    'percentage': len(cluster_members) / len(cluster_df) * 100,
                    'key_features': {}
                }

                # Identify distinguishing features (if original features available)
                if isinstance(original_features, pd.DataFrame):
                    numeric_cols = original_features.select_dtypes(include=['float64', 'int64']).columns

                    for col in numeric_cols:
                        if col in cluster_members.columns:
                            # Compare cluster mean to overall mean
                            cluster_mean = cluster_members[col].mean()
                            overall_mean = cluster_df[col].mean()

                            # Calculate standardized difference
                            if cluster_df[col].std() > 0:
                                std_diff = (cluster_mean - overall_mean) / cluster_df[col].std()

                                # Store if the difference is significant
                                if abs(std_diff) > 0.5:  # Threshold for significance
                                    cluster_analysis[method][cluster_id]['key_features'][col] = {
                                        'mean': cluster_mean,
                                        'overall_mean': overall_mean,
                                        'std_diff': std_diff
                                    }

        return cluster_analysis

    def generate_health_recommendations(self, patient_id, cluster_assignments,
                                        cluster_analysis, recommendation_rules):
        """
        Generate personalized health recommendations based on cluster membership.

        Parameters:
        -----------
        patient_id : str
            Patient ID
        cluster_assignments : pandas.DataFrame
            DataFrame with patient IDs and cluster assignments
        cluster_analysis : dict
            Analysis of cluster characteristics
        recommendation_rules : dict
            Rules for generating recommendations based on cluster characteristics

        Returns:
        --------
        dict
            Personalized recommendations
        """
        self.logger.info(f"Generating recommendations for patient {patient_id}...")

        # Get cluster assignments for this patient
        patient_row = cluster_assignments[cluster_assignments['patient_id'] == patient_id]

        if len(patient_row) == 0:
            self.logger.warning(f"Patient {patient_id} not found in cluster assignments")
            return {"error": "Patient not found"}

        hierarchical_cluster = patient_row['hierarchical_cluster'].values[0]
        gmm_cluster = patient_row['gmm_cluster'].values[0]

        # Get GMM probabilities for overlapping clusters
        gmm_probs = {}
        for i in range(self.config['num_clusters']):
            prob_col = f'gmm_prob_cluster_{i}'
            if prob_col in patient_row.columns:
                gmm_probs[i] = patient_row[prob_col].values[0]

        # Generate primary recommendations based on main cluster
        recommendations = {
            "primary_recommendations": [],
            "secondary_recommendations": [],
            "screening_recommendations": [],
            "lifestyle_recommendations": []
        }

        # Apply recommendation rules based on cluster membership
        if recommendation_rules and 'hierarchical_cluster' in recommendation_rules:
            if str(hierarchical_cluster) in recommendation_rules['hierarchical_cluster']:
                cluster_rules = recommendation_rules['hierarchical_cluster'][str(hierarchical_cluster)]

                # Add primary recommendations
                if 'primary' in cluster_rules:
                    recommendations['primary_recommendations'] = cluster_rules['primary']

                # Add screening recommendations
                if 'screenings' in cluster_rules:
                    recommendations['screening_recommendations'] = cluster_rules['screenings']

                # Add lifestyle recommendations
                if 'lifestyle' in cluster_rules:
                    recommendations['lifestyle_recommendations'] = cluster_rules['lifestyle']

        # Add secondary recommendations based on other probable clusters
        # (for GMM with significant probability but not the main cluster)
        for cluster_id, prob in gmm_probs.items():
            if cluster_id != gmm_cluster and prob > 0.3:  # Significant probability threshold
                if (recommendation_rules and 'gmm_cluster' in recommendation_rules and
                    str(cluster_id) in recommendation_rules['gmm_cluster']):

                    cluster_rules = recommendation_rules['gmm_cluster'][str(cluster_id)]

                    if 'primary' in cluster_rules:
                        # Add as secondary recommendations with probability context
                        for rec in cluster_rules['primary']:
                            recommendations['secondary_recommendations'].append({
                                "recommendation": rec,
                                "probability": f"{prob:.2f}",
                                "source_cluster": cluster_id
                            })

        # Add explanation of why these recommendations were made
        recommendations["explanation"] = {
            "main_cluster_id": hierarchical_cluster,
            "main_cluster_characteristics": {},
            "probability_distribution": gmm_probs
        }

        # Add cluster characteristics if available
        if (cluster_analysis and 'hierarchical_cluster' in cluster_analysis and
            hierarchical_cluster in cluster_analysis['hierarchical_cluster']):

            cluster_info = cluster_analysis['hierarchical_cluster'][hierarchical_cluster]

            if 'key_features' in cluster_info:
                # Add top 5 most distinctive features
                sorted_features = sorted(
                    cluster_info['key_features'].items(),
                    key=lambda x: abs(x[1]['std_diff']),
                    reverse=True
                )[:5]

                for feature, values in sorted_features:
                    recommendations["explanation"]["main_cluster_characteristics"][feature] = {
                        "direction": "higher" if values['std_diff'] > 0 else "lower",
                        "magnitude": abs(values['std_diff'])
                    }

        return recommendations

    def visualize_clusters(self, integrated_data, labels, method='hierarchical',
                          patient_ids=None, highlighted_patient=None):
        """
        Visualize the identified clusters.

        Parameters:
        -----------
        integrated_data : numpy.ndarray
            Integrated data after dimensionality reduction
        labels : numpy.ndarray
            Cluster assignments
        method : str
            Clustering method name for the plot title
        patient_ids : list, optional
            List of patient IDs for labeling specific points
        highlighted_patient : str, optional
            Patient ID to highlight in the visualization

        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with the visualization
        """
        self.logger.info(f"Visualizing clusters from {method} clustering...")

        # Apply t-SNE for visualization if data has more than 2 dimensions
        if integrated_data.shape[1] > 2:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=self.config['random_state'])
            viz_data = tsne.fit_transform(integrated_data)
        else:
            viz_data = integrated_data

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Create a scatter plot with different colors for each cluster
        scatter = plt.scatter(viz_data[:, 0], viz_data[:, 1], c=labels,
                           cmap='viridis', alpha=0.7, s=50)

        # Add a colorbar
        plt.colorbar(scatter, label='Cluster')

        # Highlight specific patient if requested
        if highlighted_patient and patient_ids:
            if highlighted_patient in patient_ids:
                idx = patient_ids.index(highlighted_patient)
                plt.scatter(viz_data[idx, 0], viz_data[idx, 1],
                           c='red', s=200, marker='*', edgecolors='black')
                plt.annotate(f"Patient: {highlighted_patient}",
                            (viz_data[idx, 0], viz_data[idx, 1]),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round", fc="w"))

        # Add labels and title
        plt.title(f'Cluster Visualization using {method.capitalize()} Clustering')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()

        return plt.gcf()

    def save_model(self, filepath):
        """
        Save the trained model and configurations.

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        self.logger.info(f"Saving model to {filepath}...")

        model_data = {
            'config': self.config,
            'pca_components': None,
            'hierarchical_model': None,
            'gmm_model': None
        }

        # Save PCA components if available
        if self.pca:
            model_data['pca_components'] = {
                'components': self.pca.components_.tolist(),
                'explained_variance': self.pca.explained_variance_.tolist(),
                'mean': self.pca.mean_.tolist()
            }

        # Save hierarchical model parameters if available
        if self.hierarchical_model:
            model_data['hierarchical_model'] = {
                'n_clusters': self.hierarchical_model.n_clusters,
                'children': self.hierarchical_model.children_.tolist() if hasattr(self.hierarchical_model, 'children_') else None
            }

        # Save GMM model parameters if available
        if self.gmm_model:
            model_data['gmm_model'] = {
                'weights': self.gmm_model.weights_.tolist(),
                'means': self.gmm_model.means_.tolist(),
                'covariances': self.gmm_model.covariances_.tolist(),
                'precisions': self.gmm_model.precisions_.tolist()
            }

        # Save encoder info
        model_data['encoders'] = {k: True for k in self.encoders.keys()}

        # Save to file
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f)

        # Save encoders separately if applicable
        for name, encoder in self.encoders.items():
            if hasattr(encoder, 'save'):
                encoder.save(f"{filepath}_{name}_encoder")

        # Save scaler
        if hasattr(self.scaler, 'mean_'):
            import joblib
            joblib.dump(self.scaler, f"{filepath}_scaler.joblib")

        # Save embedding model if available
        if self.embedding_model:
            self.embedding_model.save(f"{filepath}_embedding_model")

        self.logger.info("Model saved successfully")

    def load_model(self, filepath):
        """
        Load a previously saved model.

        Parameters:
        -----------
        filepath : str
            Path to load the model from

        Returns:
        --------
        bool
            True if model loaded successfully
        """
        self.logger.info(f"Loading model from {filepath}...")

        try:
            # Load metadata
            with open(f"{filepath}_metadata.json", 'r') as f:
                model_data = json.load(f)

            # Load configuration
            self.config = model_data['config']

            # Reconstruct PCA if available
            if model_data['pca_components']:
                from sklearn.decomposition import PCA

                self.pca = PCA(n_components=len(model_data['pca_components']['explained_variance']))
                self.pca.components_ = np.array(model_data['pca_components']['components'])
                self.pca.explained_variance_ = np.array(model_data['pca_components']['explained_variance'])
                self.pca.mean_ = np.array(model_data['pca_components']['mean'])

            # Load encoders if available
            if 'encoders' in model_data:
                for encoder_name in model_data['encoders'].keys():
                    try:
                        if encoder_name == 'clinical_notes':
                            self.encoders[encoder_name] = Doc2Vec.load(f"{filepath}_{encoder_name}_encoder")
                    except Exception as e:
                        self.logger.warning(f"Error loading encoder {encoder_name}: {e}")

            # Load scaler
            try:
                import joblib
                self.scaler = joblib.load(f"{filepath}_scaler.joblib")
            except Exception as e:
                self.logger.warning(f"Error loading scaler: {e}")
                self.scaler = StandardScaler()

            # Load embedding model if available
            try:
                self.embedding_model = tf.keras.models.load_model(f"{filepath}_embedding_model")
            except Exception as e:
                self.logger.warning(f"Error loading embedding model: {e}")

            # Reconstruct GMM model if available
            if model_data['gmm_model']:
                self.gmm_model = GaussianMixture(
                    n_components=len(model_data['gmm_model']['weights']),
                    random_state=self.config['random_state']
                )
                self.gmm_model.weights_ = np.array(model_data['gmm_model']['weights'])
                self.gmm_model.means_ = np.array(model_data['gmm_model']['means'])
                self.gmm_model.covariances_ = np.array(model_data['gmm_model']['covariances'])
                self.gmm_model.precisions_ = np.array(model_data['gmm_model']['precisions'])

            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
        
import json
import numpy as np
import pandas as pd

# Create a custom JSON encoder that handles NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def example_usage():

    # Sample data generation (in a real scenario, this would be loaded from databases)
    def generate_sample_data(num_patients=100):
        np.random.seed(42)

        # Patient IDs
        patient_ids = [f"P{i:04d}" for i in range(num_patients)]

        # Structured clinical data
        age_groups = np.random.choice(['18-30', '31-45', '46-60', '61+'], size=num_patients)
        bmi = np.random.normal(25, 5, num_patients)
        systolic_bp = np.random.normal(120, 15, num_patients)
        diastolic_bp = np.random.normal(80, 10, num_patients)
        heart_rate = np.random.normal(75, 10, num_patients)

        # Create health conditions and screening history with age-appropriate distributions
        health_conditions = []
        screening_history = []

        for age in age_groups:
            if age == '18-30':
                conditions = np.random.choice(['None', 'Anxiety', 'Depression', 'PCOS'],
                                           p=[0.7, 0.1, 0.1, 0.1])
                screening = np.random.choice(['Up to date', 'Overdue', 'Never'],
                                          p=[0.5, 0.3, 0.2])
            elif age == '31-45':
                conditions = np.random.choice(['None', 'Hypertension', 'Diabetes', 'PCOS', 'Anxiety'],
                                           p=[0.6, 0.1, 0.1, 0.1, 0.1])
                screening = np.random.choice(['Up to date', 'Overdue', 'Never'],
                                          p=[0.6, 0.3, 0.1])
            elif age == '46-60':
                conditions = np.random.choice(['None', 'Hypertension', 'Diabetes', 'Osteoporosis', 'Depression'],
                                           p=[0.5, 0.2, 0.1, 0.1, 0.1])
                screening = np.random.choice(['Up to date', 'Overdue', 'Never'],
                                          p=[0.7, 0.2, 0.1])
            else:  # 61+
                conditions = np.random.choice(['None', 'Hypertension', 'Diabetes', 'Osteoporosis', 'Cardiovascular'],
                                           p=[0.4, 0.2, 0.2, 0.1, 0.1])
                screening = np.random.choice(['Up to date', 'Overdue', 'Never'],
                                          p=[0.6, 0.3, 0.1])

            health_conditions.append(conditions)
            screening_history.append(screening)

        # Create structured data DataFrame
        structured_data = pd.DataFrame({
            'patient_id': patient_ids,
            'age_group': age_groups,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'health_condition': health_conditions,
            'screening_history': screening_history
        })
        structured_data.set_index('patient_id', inplace=True)

        # Generate clinical notes
        clinical_notes = []
        for i, patient in enumerate(patient_ids):
            age = age_groups[i]
            condition = health_conditions[i]

            if condition == 'None':
                note = f"Patient is a woman in {age} age group with no significant health issues. " \
                       f"Regular check-up shows normal vital signs. BMI is {bmi[i]:.1f}. " \
                       f"Blood pressure is {systolic_bp[i]:.0f}/{diastolic_bp[i]:.0f}. " \
                       f"Patient reports good overall health with occasional stress from work."
            else:
                note = f"Patient is a woman in {age} age group with history of {condition}. " \
                       f"Current vital signs: BMI {bmi[i]:.1f}, BP {systolic_bp[i]:.0f}/{diastolic_bp[i]:.0f}. " \
                       f"Medication compliance is good. Patient reports moderate stress levels " \
                       f"and has {screening_history[i].lower()} preventive screenings."

            clinical_notes.append(note)

        # Generate wearable device data
        wearable_data = {}
        for i, patient in enumerate(patient_ids):
            # Create 30 days of data
            num_days = 30
            timestamps = pd.date_range(start='2023-01-01', periods=num_days, freq='D')

            # Generate daily metrics with some noise and trends
            # Healthier lifestyles for those with up-to-date screenings
            activity_base = 8000 if screening_history[i] == 'Up to date' else 5000
            sleep_base = 7.5 if screening_history[i] == 'Up to date' else 6.5

            steps = np.random.normal(activity_base, 2000, num_days)
            sleep_hours = np.random.normal(sleep_base, 1, num_days)
            heart_rate_daily = np.random.normal(heart_rate[i], 5, num_days)

            # Add some weekly patterns (less activity on weekends)
            for j in range(num_days):
                if j % 7 >= 5:  # Weekend
                    steps[j] *= 0.8

            wearable_df = pd.DataFrame({
                'timestamp': timestamps,
                'steps': steps,
                'sleep_hours': sleep_hours,
                'avg_heart_rate': heart_rate_daily
            })

            wearable_data[patient] = wearable_df

        # Generate survey data
        preferences = []
        communication_prefs = []
        health_priorities = []

        for i, patient in enumerate(patient_ids):
            # Define preferences based on age group
            if age_groups[i] == '18-30':
                pref = np.random.choice(['Digital', 'In-person', 'Hybrid'], p=[0.6, 0.2, 0.2])
                comm = np.random.choice(['App', 'Email', 'Text', 'Phone'], p=[0.5, 0.3, 0.1, 0.1])
                priority = np.random.choice(['Mental Health', 'Preventive', 'Reproductive', 'Fitness'],
                                          p=[0.3, 0.2, 0.3, 0.2])
            elif age_groups[i] == '31-45':
                pref = np.random.choice(['Digital', 'In-person', 'Hybrid'], p=[0.4, 0.3, 0.3])
                comm = np.random.choice(['App', 'Email', 'Text', 'Phone'], p=[0.4, 0.3, 0.2, 0.1])
                priority = np.random.choice(['Reproductive', 'Preventive', 'Mental Health', 'Chronic Disease'],
                                          p=[0.4, 0.2, 0.2, 0.2])
            elif age_groups[i] == '46-60':
                pref = np.random.choice(['Digital', 'In-person', 'Hybrid'], p=[0.3, 0.4, 0.3])
                comm = np.random.choice(['App', 'Email', 'Text', 'Phone'], p=[0.2, 0.4, 0.1, 0.3])
                priority = np.random.choice(['Preventive', 'Chronic Disease', 'Menopause', 'Fitness'],
                                          p=[0.3, 0.3, 0.3, 0.1])
            else:  # 61+
                pref = np.random.choice(['Digital', 'In-person', 'Hybrid'], p=[0.2, 0.6, 0.2])
                comm = np.random.choice(['App', 'Email', 'Text', 'Phone'], p=[0.1, 0.3, 0.1, 0.5])
                priority = np.random.choice(['Chronic Disease', 'Preventive', 'Quality of Life', 'Mental Health'],
                                          p=[0.4, 0.3, 0.2, 0.1])

            preferences.append(pref)
            communication_prefs.append(comm)
            health_priorities.append(priority)

        # Create Likert scale responses
        satisfaction_levels = np.random.choice(['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
                                            size=num_patients)
        convenience_ratings = np.random.choice(['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
                                            size=num_patients)

        # Create survey dataframe
        survey_data = pd.DataFrame({
            'patient_id': patient_ids,
            'care_preference': preferences,
            'communication_preference': communication_prefs,
            'health_priority': health_priorities,
            'satisfaction_level': satisfaction_levels,
            'convenience_rating': convenience_ratings
        })
        survey_data.set_index('patient_id', inplace=True)

        return patient_ids, structured_data, clinical_notes, wearable_data, survey_data

    # Generate sample data
    patient_ids, structured_data, clinical_notes, wearable_data, survey_data = generate_sample_data(100)

    # Initialize the platform
    platform = MultiModalHealthPlatform(config={
        'embedding_dim': 64,
        'num_clusters': 5,
        'pca_components': 15,
        'random_state': 42,
        'auto_determine_clusters': True
    })

    # Preprocess each data source
    structured_processed = platform.preprocess_structured_data(structured_data)
    notes_embeddings = platform.preprocess_unstructured_data(clinical_notes)
    wearable_features = platform.preprocess_wearable_data(wearable_data)
    survey_processed = platform.preprocess_survey_data(survey_data)

    # Integrate all data sources
    integrated_data = platform.integrate_data_sources(
        structured_processed,
        notes_embeddings,
        wearable_features,
        survey_processed,
        patient_ids
    )

    # Apply hierarchical clustering
    hierarchical_labels = platform.apply_hierarchical_clustering(integrated_data)

    # Apply GMM clustering
    gmm_labels, gmm_probs = platform.apply_gmm_clustering(integrated_data)

    # Create DataFrame with patient IDs and cluster assignments
    cluster_assignments = pd.DataFrame({
        'patient_id': patient_ids,
        'hierarchical_cluster': hierarchical_labels,
        'gmm_cluster': gmm_labels
    })

    # Add GMM probabilities
    for i in range(platform.config['num_clusters']):
        cluster_assignments[f'gmm_prob_cluster_{i}'] = gmm_probs[:, i]

    # Analyze cluster characteristics
    original_features = pd.concat([
        structured_data,
        survey_data
    ], axis=1)

    cluster_analysis = platform.analyze_clusters(
        integrated_data,
        hierarchical_labels,
        gmm_labels,
        gmm_probs,
        original_features,
        patient_ids
    )

    # Define sample recommendation rules based on cluster analysis
    # In a real application, these would be derived from medical guidelines and expert input
    recommendation_rules = {
        'hierarchical_cluster': {
            '0': {
                'primary': [
                    "Schedule annual well-woman exam",
                    "Consider mental health screening"
                ],
                'screenings': [
                    "Pap smear every 3 years",
                    "STI screening"
                ],
                'lifestyle': [
                    "Stress management techniques",
                    "Regular physical activity"
                ]
            },
            '1': {
                'primary': [
                    "Monitor blood pressure regularly",
                    "Schedule diabetes screening"
                ],
                'screenings': [
                    "Mammogram",
                    "Bone density test"
                ],
                'lifestyle': [
                    "Heart-healthy diet",
                    "Regular physical activity"
                ]
            },
            '2': {
                'primary': [
                    "Regular reproductive health check-ups",
                    "Fertility counseling if planning pregnancy"
                ],
                'screenings': [
                    "Pap smear",
                    "HPV testing"
                ],
                'lifestyle': [
                    "Prenatal vitamins if planning pregnancy",
                    "Balanced nutrition"
                ]
            },
            '3': {
                'primary': [
                    "Chronic disease management",
                    "Medication review"
                ],
                'screenings': [
                    "Blood pressure monitoring",
                    "A1C testing"
                ],
                'lifestyle': [
                    "Mediterranean diet",
                    "Low-impact exercise"
                ]
            },
            '4': {
                'primary': [
                    "Preventive screening schedule",
                    "Age-appropriate vaccinations"
                ],
                'screenings': [
                    "Mammogram",
                    "Colorectal cancer screening"
                ],
                'lifestyle': [
                    "Weight management",
                    "Bone-strengthening exercises"
                ]
            }
        },
        'gmm_cluster': {
            '0': {
                'primary': [
                    "Mental health support",
                    "Stress management resources"
                ]
            },
            '1': {
                'primary': [
                    "Cardiovascular health monitoring",
                    "Blood pressure management"
                ]
            },
            '2': {
                'primary': [
                    "Family planning resources",
                    "Fertility support if needed"
                ]
            },
            '3': {
                'primary': [
                    "Chronic disease management plan",
                    "Medication optimization"
                ]
            },
            '4': {
                'primary': [
                    "Comprehensive preventive care",
                    "Healthy aging resources"
                ]
            }
        }
    }

    # Generate recommendations for a specific patient
    sample_patient_id = patient_ids[0]
    recommendations = platform.generate_health_recommendations(
        sample_patient_id,
        cluster_assignments,
        cluster_analysis,
        recommendation_rules
    )

    # Visualize the clusters
    hierarchical_viz = platform.visualize_clusters(
        integrated_data,
        hierarchical_labels,
        method='hierarchical',
        patient_ids=patient_ids,
        highlighted_patient=sample_patient_id
    )

    # Save the visualization
    hierarchical_viz.savefig('hierarchical_clusters.png')

    # Print sample patient recommendations
    print(f"\nRecommendations for patient {sample_patient_id}:")
    print(json.dumps(recommendations, indent=2, cls=NumpyEncoder))

    # Save the model
    platform.save_model('women_health_platform_model')

    # Summary of clusters
    print("\nCluster Analysis Summary:")
    for method in ['hierarchical_cluster', 'gmm_cluster']:
        print(f"\n{method.replace('_', ' ').title()}:")
        for cluster_id, info in cluster_analysis[method].items():
            print(f"  Cluster {cluster_id}: {info['size']} patients ({info['percentage']:.1f}%)")

            # Print top 3 distinguishing features
            if 'key_features' in info:
                sorted_features = sorted(
                    info['key_features'].items(),
                    key=lambda x: abs(x[1]['std_diff']),
                    reverse=True
                )[:3]

                if sorted_features:
                    print("  Key characteristics:")
                    for feature, values in sorted_features:
                        direction = "higher" if values['std_diff'] > 0 else "lower"
                        print(f"    - {feature}: {direction} than average (diff: {values['std_diff']:.2f})")

    return platform, integrated_data, hierarchical_labels, gmm_labels, cluster_assignments, recommendations

# For real-world deployment, you would add functions like:
#  - process_new_patient(platform, patient_data)
#  - schedule_model_retraining(platform, new_data_threshold=1000)
#  - integration_with_ehr_systems()
#  - recommendation_feedback_loop()
#  - compliance_and_audit_logging()

if __name__ == "__main__":
    platform, data, h_labels, g_labels, assignments, recs = example_usage()
    print("Platform implementation complete and demonstrated with sample data.")