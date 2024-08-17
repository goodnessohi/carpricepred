import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline, CustomData  # Import your classes

# Create a title
st.title("Car Price Prediction App")

# Create input fields
year = st.number_input("Year", min_value=1900, max_value=2024)
km_driven = st.number_input("Kilometers Driven", min_value=0)
mileage = st.number_input("Mileage", min_value=0.0)
engine = st.number_input("Engine", min_value=0.0)
seats = st.number_input("Seats", min_value=1)
max_power = st.number_input("Max Power", min_value=0.0)
fuel = st.selectbox("Fuel", ['Diesel', 'Petrol', 'LPG', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
brand = st.selectbox("Brand", ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 
                               'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet', 
                               'Fiat', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 
                               'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 
                               'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Force', 'Ambassador', 
                               'Ashok', 'Isuzu', 'Opel', 'Peugeot'])

# Create a button to submit inputs
if st.button("Submit"):
    # Create a CustomData object with the inputs
    data = CustomData(
        year=year,
        km_driven=km_driven,
        mileage=mileage,
        engine=engine,
        seats=seats,
        max_power=max_power,
        fuel=fuel,
        seller_type=seller_type,
        transmission=transmission,
        owner=owner,
        brand=brand
    )
    
    # Convert CustomData to DataFrame
    df = data.get_data_as_df()
    
    # Create a PredictPipeline object
    pipeline = PredictPipeline()
    
    # Create a progress bar
    with st.spinner('Computing prediction...'):
    # Get the prediction
        prediction = pipeline.predict(df)

    # Truncate decimal points and add dollar sign
        predicted_price = "${:,.2f}".format(prediction[0] / 83.88)

    
    # Display the inputs and prediction
    st.write("Inputs:")
    st.write(df)
    st.write("Prediction:")
    st.write(predicted_price)

