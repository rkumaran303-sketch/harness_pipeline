import os
from mangum import Mangum
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import pymongo
from fastapi import FastAPI, HTTPException, Query, Depends, Form, status
from dotenv import load_dotenv
import logging
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
import traceback
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from typing import Optional
from jose import jwt, JWTError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("API_KEY")
RECORDER_LIST_URL = os.getenv("RECORDER_LIST_URL")
EVENTS_API_URL = os.getenv("EVENTS_API_URL")
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
DATABASE_ALLEVENTS_COLLECTION_NAME = os.getenv("DATABASE_ALLEVENTS_COLLECTION_NAME")
DATABASE_LOG_COLL_NAME = os.getenv("DATABASE_LOG_COLL_NAME")
DIFFERENCE_BETWEEN_DATE = int(os.environ.get('DIFFERENCE_BETWEEN_DATE'))

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3_client = boto3.client("s3", region_name=AWS_REGION)

if not all([DATABASE_URL, DATABASE_NAME, DATABASE_LOG_COLL_NAME]):
    raise RuntimeError("Missing one or more required environment variables")

# MongoDB setup
client = pymongo.MongoClient(DATABASE_URL)
database = client[DATABASE_NAME]
log_collection = database[DATABASE_LOG_COLL_NAME]
log_collection.create_index([("imei", 1), ("tsInMilliSeconds", 1)])

# ---------------------
# JWT Token Authentication
# ---------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/device/auth")
VALID_USERNAME = os.getenv("AUTH_USERNAME")
VALID_PASSWORD = os.getenv("AUTH_PASSWORD")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES"))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username != VALID_USERNAME:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"message": "Invalid token user"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"username": username}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Invalid or expired token"},
            headers={"WWW-Authenticate": "Bearer"},
        )

# ---------------------
# Utility functions
# ---------------------
def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def validate_date_range(start_date: str, end_date: str):
    try:
        start_date_ts = int(datetime.fromisoformat(start_date.replace("Z", "+00:00")).timestamp() * 1000)
        end_date_ts = int(datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DDThh:mm:ssZ")
    
    if start_date_ts >= end_date_ts:
        raise HTTPException(status_code=400, detail="Start date must be earlier than end date")
    
    date_diff = (end_date_ts - start_date_ts) / (1000 * 60 * 60 * 24)
    if date_diff > DIFFERENCE_BETWEEN_DATE:
        raise HTTPException(status_code=422, detail=f"The date range must not exceed {DIFFERENCE_BETWEEN_DATE} days")

    return start_date_ts, end_date_ts

def parse_iso_z_timestamp(ts):
    """Safely parse ISO8601 timestamps ending with Z (UTC) with up to 6 microseconds digits."""
    if not isinstance(ts, str):
        return None
    ts = ts.replace("Z", "+00:00")
    if "." in ts:
        main, rest = ts.split(".", 1)
        if "+" in rest:
            frac, tz = rest.split("+", 1)
            frac = frac[:6]
            ts = f"{main}.{frac}+{tz}"
        else:
            frac = rest[:6]
            ts = f"{main}.{frac}"
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def get_device_metadata(api_url: str, api_key: str):
    headers = {"APIKey": api_key}
    try:
        logger.info(f"Fetching device metadata from: {api_url}")
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        devices = [
            {
                "RecorderID": d.get("RecorderID"),
                "VIN": d.get("VIN"),
                "TSPCustomerIDDV": d.get("TSPCustomerIDDV")
            }
            for d in data 
        ]
        logger.info(f"Found {len(devices)} devices")
        return devices
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch device metadata: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in get_device_metadata: {e}")
        return []

def plot_event_deltas(recorder_id, deltas, short_repeat_count, output_folder, threshold=30):
    if not deltas:
        return

    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            15, 20, 30, 45, 60, 90, 120, 180,
            300, 600, 900, 999999]

    tick_labels = [str(int(b)) for b in bins[:-2]] + ['900', '900+']

    plt.figure(figsize=(12, 5))
    plt.hist(deltas, bins=bins, edgecolor='black', align='left')

    plt.title(f"Time Gaps Between Events: {recorder_id}")
    plt.xlabel("Delta (seconds)")
    plt.ylabel("Frequency")
    plt.xticks(bins[:-1], tick_labels, rotation=45)

    label = f"Short repeats (≤ {threshold}s): {short_repeat_count}"
    plt.text(
        0.95, 0.95, label,
        ha='right', va='top', transform=plt.gca().transAxes,
        fontsize=9, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
    )

    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"{recorder_id}_histogram.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {plot_path}")

def check_delivery_order(current_event, previous_event):
    try:
        current_id = current_event.get("EventLogID")
        prev_id = previous_event.get("EventLogID")

        if not current_id or not prev_id:
            return {"time(diff bwt del)": None, "OutOfSequence": "No"}

        current_doc = database[DATABASE_ALLEVENTS_COLLECTION_NAME].find_one(
            {"vendorEventId": current_id}, {"createdAtMs": 1}
        )
        prev_doc = database[DATABASE_ALLEVENTS_COLLECTION_NAME].find_one(
            {"vendorEventId": prev_id}, {"createdAtMs": 1}
        )

        current_delivery = current_doc.get("createdAtMs") if current_doc else None
        prev_delivery = prev_doc.get("createdAtMs") if prev_doc else None

        if current_delivery is not None and prev_delivery is not None:
            delivery_time_diff = current_delivery - prev_delivery
            out_of_sequence = "Yes" if delivery_time_diff < 0 else "No"
            return {
                "time(diff bwt del)": delivery_time_diff,
                "OutOfSequence": out_of_sequence,
            }
        else:
            return {"time(diff bwt del)": None, "OutOfSequence": "No"}

    except Exception as e:
        logger.error(f"Database query error in check_delivery_order: {e}")
        return {"time(diff bwt del)": None, "OutOfSequence": "No"}

def fetch_events_and_save_excel(events_api_url, api_key, devices, from_to, imei):
    headers = {"APIKey": api_key}

    try:
        analysis_dir = "/tmp/analysis"
        os.makedirs(analysis_dir, exist_ok=True)

        summary_rows, device_sheets = [], {}

        for device in devices:
            recorder_id = device["RecorderID"]
            vin = device.get("VIN", "")
            customer_id = device.get("TSPCustomerIDDV", "")

            all_events = []
            for event_type in ["IgnitionOn", "IgnitionOff"]:
                params = {**from_to, "RecorderID": recorder_id, "EventType": event_type}
                try:
                    r = requests.get(events_api_url, headers=headers, params=params, timeout=30)
                    r.raise_for_status()
                    for item in r.json():
                        all_events.append({
                            "EventLogID": item.get("EventLogID"),
                            "EventTypeIDDV": item.get("EventTypeIDDV"),
                            "EventDatetime": item.get("EventDateTime")
                        })
                except requests.RequestException as e:
                    logger.error(f"Failed to fetch {event_type} for {recorder_id}: {e}")

            if not all_events:
                continue

            all_events.sort(key=lambda x: x["EventDatetime"] or "")

            event_ids = [e["EventLogID"] for e in all_events if e.get("EventLogID")]
            delivery_docs = list(
                database[DATABASE_ALLEVENTS_COLLECTION_NAME].find(
                    {"vendorEventId": {"$in": event_ids}},
                    {"vendorEventId": 1, "createdAtStr": 1}
                )
            )

            delivery_map = {}
            for d in delivery_docs:
                created_at_str = d.get("createdAtStr")
                if created_at_str:
                    try:
                        created_at_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        formatted_str = created_at_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        delivery_map[d["vendorEventId"]] = formatted_str
                    except Exception as e:
                        logger.warning(f"Failed to parse {created_at_str}: {e}")
                        delivery_map[d["vendorEventId"]] = created_at_str

            prev_event = None
            SHORT_DELTA_THRESHOLD = 60
            out_of_sequence_count = short_repeat_count = missing_pair_count = 0
            count_of_undelivered_events = 0
            for event in all_events:
                delivery_str = delivery_map.get(event["EventLogID"])
                event["EventDeliveryDatetime"] = delivery_str

                event_dt = parse_iso_z_timestamp(event.get("EventDatetime"))
                delivery_dt = parse_iso_z_timestamp(delivery_str)

                if prev_event:
                    prev_event_dt = parse_iso_z_timestamp(prev_event.get("EventDatetime"))
                    if prev_event_dt and event_dt:
                        delta_gen = (event_dt - prev_event_dt).total_seconds()
                        event["DeltaEventDatetime"] = round(delta_gen, 2)
                    else:
                        delta_gen = None
                        event["DeltaEventDatetime"] = ""
                else:
                    delta_gen = None
                    event["DeltaEventDatetime"] = ""

                if prev_event:
                    prev_delivery_dt = parse_iso_z_timestamp(prev_event.get("EventDeliveryDatetime"))
                    if prev_delivery_dt and delivery_dt:
                        delta_del = (delivery_dt - prev_delivery_dt).total_seconds()
                        event["DeltaEventDeliveryDatetime"] = round(delta_del, 2)
                    else:
                        delta_del = None
                        event["DeltaEventDeliveryDatetime"] = ""
                else:
                    delta_del = None
                    event["DeltaEventDeliveryDatetime"] = ""

                if delta_gen is not None and delta_gen <= SHORT_DELTA_THRESHOLD:
                    event["IgnitionFlicker"] = "Yes"
                    short_repeat_count += 1
                else:
                    event["IgnitionFlicker"] = "No"

                if delta_del is not None and delta_del < 0:
                    event["OutOfSequence"] = "Yes"
                    out_of_sequence_count += 1
                else:
                    event["OutOfSequence"] = "No"

                if prev_event and event["EventTypeIDDV"] == prev_event["EventTypeIDDV"]:
                    event["Unpaired"] = "Yes"
                    missing_pair_count += 1
                else:
                    event["Unpaired"] = "No"

                event["Undelivered"] = "Yes" if not delivery_str else "No"
                if event["Undelivered"] == "Yes":
                    count_of_undelivered_events += 1
                prev_event = event

            df = pd.DataFrame(all_events, columns=[
                "EventLogID",
                "EventTypeIDDV",
                "EventDatetime",
                "EventDeliveryDatetime",
                "DeltaEventDatetime",
                "DeltaEventDeliveryDatetime",
                "IgnitionFlicker",
                "OutOfSequence",
                "Unpaired",
                "Undelivered"
            ])
            device_sheets[recorder_id] = df

            summary_rows.append({
                "RecorderID": recorder_id,
                "VIN": vin,
                "TSPCustomerIDDV": customer_id,
                "NumEvents": len(all_events),
                "IgnitionFlicker": short_repeat_count,
                "Unpaired": missing_pair_count,
                "OutOfSequence": out_of_sequence_count,
                "Undelivered" : count_of_undelivered_events
            })

        # If no data at all, add empty summary row
        if not summary_rows:
            summary_rows.append({
                "RecorderID": recorder_id,
                "VIN": vin,
                "TSPCustomerIDDV": customer_id,
                "NumEvents": 0,
                "IgnitionFlicker": 0,
                "Unpaired": 0,
                "OutOfSequence": 0,
                "Undelivered": 0
            })
            
        excel_filename = f"device_data_{imei}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        summary_df = pd.DataFrame(summary_rows)

        try:
            import xlsxwriter  # noqa: F401
            engine = "xlsxwriter"
        except ImportError:
            engine = "openpyxl"

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine=engine) as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            for rid, df in device_sheets.items():
                sheet_name = str(rid)[:31]
                df.to_excel(writer, index=False, sheet_name=sheet_name)

        buffer.seek(0)

        s3_key = f"reports/{excel_filename}"
        s3_client.upload_fileobj(buffer, S3_BUCKET, s3_key)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=1800  # 30 minutes
        )

        logger.info(f"Excel uploaded to S3. Pre-signed URL generated: {presigned_url}")
        return presigned_url

    except Exception as e:
        logger.error(f"Error in fetch_events_and_save_excel: {e}")
        raise

# ---------------------
# Response Models
# ---------------------
class ReportResponse(BaseModel):
    message: str
    imei: Optional[str] = None
    url: Optional[str] = None
    status: Optional[str] = None

class ErrorResponse(BaseModel):
    message: str

# ---------------------
# FastAPI app & endpoints
# ---------------------
app = FastAPI(
    title="Device Events API's",
    description="This is a collection of API's which will fetch, discard and reprocess events",
    version="1.0.0",
    contact={
        "name": "API Team",
        "email": "support@example.com"
    },
    docs_url="/device/docs",
    # openapi_url="/device/openapi.json"
)


@app.post("/device/auth", tags=["Authentication"], summary="Generate Bearer Token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate using username and password to receive a JWT Bearer token.
    """
    if form_data.username == VALID_USERNAME and form_data.password == VALID_PASSWORD:
        access_token = create_access_token({"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"message": "Invalid username or password"})

@app.get(
    "/device/data/xirgo/kp2",
    tags=["Device Data Ignition Events API"],
    summary="Generate and download device data report",
    description="Download a report of anomalous ignition events for a given IMEI and time range.\nPlease ensure that the date range is less than or equal to two days",
    response_model=ReportResponse,
    responses={
        200: {
            "description": "Report generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Report generated successfully. Please download within 30 minutes after URL generation",
                        "imei": "Requested IMEI number",
                        "url": "File download URL"
                    }
                }
            }
        },
        404: {
            "model": ErrorResponse,
            "description": "No device found",
            "content": {"application/json": {"example": {"message": "No Device Found"}}}
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid input or date range",
            "content": {"application/json": {"example": {"message": "Invalid date format"}}}
        },
        422: {
            "model": ErrorResponse,
            "description": "Validation Error",
            "content": {"application/json": {"example": {"message": "Date Range Validation Error"}}}
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error",
            "content": {"application/json": {"example": {"message": "Something went wrong"}}}
        }
    }
)
def adapter(
    imei: str = Query(..., description="The IMEI of the device."),
    start_time: str = Query(..., description="Start time in ISO format UTC (e.g., YYYY-MM-DDThh:mm:ss)."),
    end_time: str = Query(..., description="End time in ISO format UTC (e.g., YYYY-MM-DDThh:mm:ss)."),
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Processing request for IMEI: {imei}, Start: {start_time}, End: {end_time}")
        start_date_ts, end_date_ts = validate_date_range(start_time, end_time)

        existing_log = log_collection.find_one({"imei": imei, "startTime": start_time, "endTime": end_time})
        if existing_log:
            logger.info(f"Found existing log for {imei}")

            # Only update the lastUpdated field, keeping createdAt and completedAt fixed
            log_collection.update_one(
                {"_id": existing_log["_id"]},
                {"$set": {"lastUpdated": utc_now_iso()}}
            )

            s3_url = existing_log.get("fileUrl")
            if s3_url:
                try:
                    # Extract S3 key and regenerate presigned URL
                    key = s3_url.split(f".amazonaws.com/")[1]
                    presigned_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': S3_BUCKET, 'Key': key},
                        ExpiresIn=1800  # 30 minutes
                    )

                    return {
                        "message": "Data processing already exists",
                        "imei": imei,
                        "url": presigned_url,
                        "status": existing_log.get("status", "Completed")
                    }
                except Exception as e:
                    logger.error(f"Failed to generate presigned URL from stored S3 URL: {e}")
                    raise HTTPException(status_code=500, detail={"message": "Failed to generate presigned URL"})

            return {
                "message": "Data processing already exists but no file found",
                "imei": imei,
                "url": None,
                "status": existing_log.get("status", "Unknown")
            }

        # If no existing log, start processing
        log_entry = {
            "imei": imei,
            "startTime": start_time,
            "endTime": end_time,
            "status": "Processing",
            "createdAt": utc_now_iso(),
            "tsInMilliSeconds": int(datetime.now(timezone.utc).timestamp() * 1000)
        }
        result = log_collection.insert_one(log_entry)
        logger.info(f"Created log entry with ID: {result.inserted_id}")

        devices = get_device_metadata(RECORDER_LIST_URL, API_KEY)
        filtered_devices = [d for d in devices if d["RecorderID"] == imei]

        if not filtered_devices:
            log_collection.update_one(
                {"imei": imei, "startTime": start_time, "endTime": end_time},
                {"$set": {"status": "No Device Found", "lastUpdated": utc_now_iso()}}
            )
            return {"message": "No Device Found", "imei": imei}

        # Generate Excel and upload to S3
        presigned_url = fetch_events_and_save_excel(
            EVENTS_API_URL, API_KEY, filtered_devices,
            {"FromTime": start_time, "ToTime": end_time},
            imei
        )

        # Extract original S3 URL (non-signed)
        s3_key = presigned_url.split(f".amazonaws.com/")[1].split("?")[0]
        original_s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

        # Update DB with permanent S3 URL
        log_collection.update_one(
            {"imei": imei, "startTime": start_time, "endTime": end_time},
            {"$set": {
                "status": "Completed",
                "fileUrl": original_s3_url,
                "completedAt": utc_now_iso(),
                "lastUpdated": utc_now_iso()
            }}
        )

        logger.info(f"Processing completed successfully for {imei}")
        return {
            "message": "Report generated successfully. Please download within 30 minutes after URL generation.",
            "imei": imei,
            "url": presigned_url,
            "status": "Completed"
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Internal error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"message": "Internal server error"})

handler = Mangum(app)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
