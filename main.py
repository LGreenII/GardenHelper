import streamlit as st
import pandas as pd
import langchain_helper as lch

# Create a DataFrame from the data
# Define the Streamlit app
def main():

    st.title('Garden Helper 🥑🌽🍎 ')
    st.subheader('Note: You only have to create the Knowledge base on the initial run of the app.')
    update_database = st.button("Create Knowledge Base")
    if update_database:
        lch.create_vector_db()
        st.write("Knowledge Base has been updated!")

    question = st.sidebar.text_input("Enter a Question:")
    if st.sidebar.button("Submit Question"):
        chain = lch.get_qa_chain()
        response = chain(question)

        st.header("Answer: ")
        st.write(response["result"])

    # Sidebar for entering zip code
    zip_code = st.sidebar.text_input('Enter Zip Code:')
    if st.sidebar.button('Submit Zip Code'):
        st.write('Hardiness Zone and Germination Time for Various Fruit by Zip Code')
        st.write(lch.usda_hardiness_zones(zip_code))
        if st.sidebar.button('Create Map'):
            st.sidebar.write("Coming Soon!")
            st.map(lch.get_map_data_frame(zip_code), latitude='lat', longitude='long', size=500, color='color')
            st.pydeck_chart(
                viewport={
                    'latitude': lch.get_map_data_frame(zip_code)['lat'],
                    'longitude': lch.get_map_data_frame(zip_code)['long'],
                    'zoom': 4
                },
                layers=[{
                    'type': 'ScatterplotLayer',
                    'data': lch.get_map_data_frame(zip_code),
                    'radiusScale': 250,
                    'radiusMinPixels': 5,
                    'getFillColor': [248, 24, 148],
                }]
            )

# Run the app
if __name__ == '__main__':
    main()
    #lch.get_map_data_frame(35749)