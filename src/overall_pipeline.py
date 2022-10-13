import ocr_pipeline
import onet_classification
import skills_analysis
import topic_modeling_top2vec

# Main Function
def main(image_path, data_path, occ_path, alt_path, tech_path, tags_path, ocr_output_path, lockdown_date_range, embedding_model, onet_corpus_path, matches_path, text_column_name):

    ocr_df = ocr_pipeline.main(image_path, ocr_output_path)
    matches = onet_classification.main(data_path, occ_path, alt_path, tech_path, tags_path, ocr_output_path, lockdown_date_range, onet_corpus_path, matches_path)

    skills_analysis.main(ocr_output_path)
    topic_modeling_top2vec.main(ocr_output_path, text_column_name, embedding_model)

    return (ocr_df, matches) #, matches)

if __name__ == '__main__':
    # Path to the folder where project data is saved
    folder_path = 'C:/Users/DELL/Documents/LIRNEasia/ADB/' #'/content/drive/MyDrive/LIRNEasia/ADB Project/'

    # Path to TopJobs advertisement images
    image_path = "C:/Users/DELL/Documents/LIRNEasia/ADB/small_batch" #"D:/nlp/top_jobs_cs_20_21/part_1/part_1b"

    # Path to the full Topjobs dataset
    data_path = folder_path+'data/data_full.xlsx'

    # Path to the dataset of ONET occupation titles
    occ_path = folder_path + 'data/onet_occ_titles.txt'

    # Path to the dataset of ONET alternate occupation titles
    alt_path = folder_path+'data/onet_alt_titles.txt'

    # Path to the dataset of technologies associated with ONET occupations
    tech_path = folder_path+'data/onet_tech_skills.txt'

    # Path to the dataset of manually annotated tags for the Topjobs data sample
    tags_path = folder_path+'data/cs_sample_tags.csv'

    # Path to the OCR outputs for the Topjobs data sample
    ocr_output_path = folder_path+'small_batch.csv' #p2a_1

    # Date range during which lockdown was implemented (ASSUMPTION: All dates beyond 2020-03-01 are considered to be under lockdown)
    lockdown_date_range = ['2020-03-01', '2022-12-31']

    # Path to ONET Corpus
    onet_corpus_path = folder_path+'data/outputs/onet_corpus.csv'

    # Path to matches file
    matches_path = folder_path+'data/outputs/sample_matches.csv'

    # Embedding Model for Top2Vec topic modelling
    embedding_model = 'universal-sentence-encoder'

    # Text Column used for Top2Vec topic modelling
    text_column_name = 'clean_text'

    main(image_path, data_path, occ_path, alt_path, tech_path, tags_path, ocr_output_path, lockdown_date_range, embedding_model, onet_corpus_path, matches_path, text_column_name)