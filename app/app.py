```python
import streamlit as st

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Om Trading Company",
    page_icon="🌾",
    layout="wide"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("🌾 Om Trading Company")
st.subheader("VPO Datta, Hansi")

st.markdown("---")

# -------------------------------
# SIDEBAR
# -------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Products", "Gallery", "Place Order", "Contact"]
)

# -------------------------------
# HOME
# -------------------------------
if menu == "Home":

    st.header("Welcome to Om Trading Company")

    st.write("""
    Om Trading Company deals in quality agricultural products
    including wheat, rice, maize, mustard, barley and other grains.

    We provide reliable procurement and supply services to
    farmers, vendors, wholesalers and retailers.
    """)

    st.image(
        "https://images.unsplash.com/photo-1500937386664-56d1dfef3854",
        use_container_width=True
    )

# -------------------------------
# PRODUCTS
# -------------------------------
elif menu == "Products":

    st.header("Our Products")

    products = {
        "Wheat": 2500,
        "Rice": 3200,
        "Mustard": 5500,
        "Maize": 2100,
        "Barley": 2400
    }

    for product, price in products.items():
        st.success(f"{product} - ₹{price} per quintal")

# -------------------------------
# GALLERY
# -------------------------------
elif menu == "Gallery":

    st.header("Photo & Video Gallery")

    st.subheader("Upload Photos")

    image_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        st.image(image_file, use_container_width=True)

    st.subheader("Upload Videos")

    video_file = st.file_uploader(
        "Choose a Video",
        type=["mp4", "mov", "avi"]
    )

    if video_file:
        st.video(video_file)

# -------------------------------
# PLACE ORDER
# -------------------------------
elif menu == "Place Order":

    st.header("Place Your Order")

    with st.form("order_form"):

        customer_name = st.text_input("Customer Name")

        mobile = st.text_input("Mobile Number")

        product = st.selectbox(
            "Select Product",
            ["Wheat", "Rice", "Mustard", "Maize", "Barley"]
        )

        quantity = st.number_input(
            "Quantity (Quintals)",
            min_value=1
        )

        address = st.text_area("Delivery Address")

        submit = st.form_submit_button("Place Order")

    if submit:

        st.success("Order Submitted Successfully!")

        st.write("### Order Details")

        st.write("Customer:", customer_name)
        st.write("Mobile:", mobile)
        st.write("Product:", product)
        st.write("Quantity:", quantity)
        st.write("Address:", address)

        st.info("""
        SMS integration requires a service like
        MSG91 or Twilio API.
        """)

# -------------------------------
# CONTACT
# -------------------------------
elif menu == "Contact":

    st.header("Contact Us")

    st.write("🏢 Company: Om Trading Company")
    st.write("📍 Address: VPO Datta, Hansi")
    st.write("📧 Email: chughmayank71@gmail.com")
    st.write("📞 Mobile: 9253588401")

    st.markdown("---")

    st.subheader("Send Inquiry")

    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    msg = st.text_area("Message")

    if st.button("Send Inquiry"):
        st.success("Inquiry Submitted Successfully!")