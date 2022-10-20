import ocr_pipeline
import onet_classification
import skills_analysis
import topic_modeling_top2vec
import yaml

# Main Function
def main(image_path, data_path, occ_path, alt_path, tech_path, ocr_output_path, lockdown_date_range, embedding_model, onet_corpus_path, matches_path, text_column_name, wordclouds_path, ocr_model_path):

    ocr_df = ocr_pipeline.main(image_path, ocr_output_path, ocr_model_path)
    matches = onet_classification.main(data_path, occ_path, alt_path, tech_path, ocr_output_path, lockdown_date_range, onet_corpus_path, matches_path)

    skills_analysis.main(ocr_output_path)
    topic_modeling_top2vec.main(ocr_output_path, wordclouds_path, text_column_name, embedding_model)

    return (ocr_df, matches)

if __name__ == '__main__':
    # Reading config.yaml
    with open("config.yaml", 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # Path to TopJobs advertisement images
    image_path = config_dict.get("image_path")

    # Path to the full Topjobs dataset
    data_path = config_dict.get("data_path")

    # Path to the dataset of ONET occupation titles
    occ_path = config_dict.get("occ_path")

    # Path to the dataset of ONET alternate occupation titles
    alt_path = config_dict.get("alt_path")

    # Path to the dataset of technologies associated with ONET occupations
    tech_path = config_dict.get("tech_path")

    # Path to the dataset of manually annotated tags for the Topjobs data sample
    tags_path = config_dict.get("tags_path")

    # Path to the OCR outputs for the Topjobs data sample
    ocr_output_path = config_dict.get("ocr_output_path")

    # Date range during which lockdown was implemented (ASSUMPTION: All dates beyond 2020-03-01 are considered to be under lockdown)
    lockdown_date_range = config_dict.get("lockdown_date_range")

    # Path to ONET Corpus
    onet_corpus_path = config_dict.get("onet_corpus_path")

    # Path to matches file
    matches_path = config_dict.get("matches_path")

    # Path to wordclouds file
    wordclouds_path = config_dict.get("wordclouds_path")

    # Path to the OCR model
    ocr_model_path = config_dict.get("ocr_model_path")

    # Embedding Model for Top2Vec topic modelling
    embedding_model = config_dict.get("embedding_model")

    # Text Column used for Top2Vec topic modelling
    text_column_name = config_dict.get("text_column_name")

    main(image_path, data_path, occ_path, alt_path, tech_path, tags_path, ocr_output_path, lockdown_date_range, embedding_model, onet_corpus_path, matches_path, text_column_name, wordclouds_path, ocr_model_path)