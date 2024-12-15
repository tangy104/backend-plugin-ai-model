import base64, io
from dotenv import load_dotenv
load_dotenv()

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    base64_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return base64_str