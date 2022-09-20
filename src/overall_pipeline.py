import ocr_pipeline
import onet_classification
import skills_analysis
import topic_modeling_top2vec

# Main Function
def main(folder_path, image_path, data_path, occ_path, alt_path, tech_path, tags_path, ocr_output_path, lockdown_date_range, embedding_model):

    ocr_df = ocr_pipeline.main(image_path, ocr_output_path)
    matches = onet_classification.main(folder_path, data_path, occ_path, alt_path, tech_path, tags_path, ocr_output_path, lockdown_date_range)

    skills_analysis.main(ocr_output_path)
    topic_modeling_top2vec.main(data_path, tags_path, ocr_output_path, embedding_model)

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

    # Embedding Model for Top2Vec topic modelling
    embedding_model = 'universal-sentence-encoder'

    main(folder_path, image_path, data_path, occ_path, alt_path, tech_path, tags_path, ocr_output_path, lockdown_date_range, embedding_model)