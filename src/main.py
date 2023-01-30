import logging
import ocr_pipeline
import onet_classification
import onet_evaluation
import skills_analysis
import topic_modeling_top2vec
import argparse

from utils import read_config_file

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler('adb_pipeline_log.log', mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def run(config_file_path):
    """ Runs the entire pipeline """

    logging.info('Reading the config file')
    config = read_config_file(config_file_path)
   
    # logging.info('Running the OCR pipeline')

    # ocr_df = ocr_pipeline.main(
    #     read_path=config['image_path'],
    #     save_path=config['ocr_output_path'],
    #     ocr_model_path=config['ocr_model_path'],
    #     acc_threshold=config['accuracy_threshold']
    # )    

    #logging.info('OCR pipeline completed successfully! Starting classifying images to onet categories')

    # matches = onet_classification.main(
    #     data_path=config['data_path'],
    #     occ_path=config['occ_path'],
    #     alt_path=config['alt_path'],
    #     tech_path=config['tech_path'],
    #     ocr_output_path=config['ocr_output_path'],
    #     lockdown_date_range=config['lockdown_date_range'],
    #     # embedding_model=config['embedding_model'],
    #     onet_corpus_path=config['onet_corpus_path'],
    #     matches_path=config['matches_path']
    # )

    # matches, confusion_matrix = onet_evaluation.main(
    #     matches_path=config['matches_path'],
    #     tags_path=config['tags_path']
    # )

    # logging.info('Classification into ONET categories completed! Starting the skills analyses scripts')


    # skills_analysis.main(
    #     ocr_output_path=config['ocr_output_path']
    # )

    topic_modeling_top2vec.main(
        ocr_output_path=config['ocr_output_path'], 
        wordclouds_path=config['wordclouds_path'], 
        text_column_name=config['text_column_name'], 
        embedding_model=config['embedding_model'],
        topic_model_path=config['topic_model_path']
    )

    # return ocr_df, matches, confusion_matrix


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--configfile",
        type=str,
        help="Path to the configuration file (required)",
        required=True
    )

    args = arg_parser.parse_args()

    run(args.configfile)

