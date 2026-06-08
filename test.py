import dashscope
import numpy as np
from http import HTTPStatus

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

texts = [
    '随着网络基础设施不断完善，越来越多偏远地区的学校开始接入数字化教学体系。原本受地域限制而难以获得优质课程的学生，如今可以通过在线课堂、直播教学和数字学习平台接触来自不同地区的优秀教师与丰富课程内容。技术手段打破了传统教育资源分布不均的问题，使学生的学习机会不再完全取决于出生地和学校条件，为更多孩子打开了接触知识和实现个人发展的通道。',
    '推动社会发展不仅需要经济增长，也需要让不同群体拥有相对均衡的发展起点。在教育领域，应持续改善区域之间在师资水平、教学内容和学习环境上的差异，通过跨区域协作、资源共享和长期投入，让来自不同家庭和地区背景的青少年都能够获得高质量培养。只有当成长机会更加普遍地覆盖各类人群时，教育才能真正发挥促进社会流动与长期进步的作用。'
]
# Similarity: 0.5334627798480506
texts = ['数字技术帮助偏远地区儿童获得优质课程','减少家庭背景对个人成长机会的影响']
# Similarity: 0.3919541952073226
# texts = ['教育公平','机会均等']
# Similarity: 0.6906505565304736

texts = ['教育公平','数字普惠']
# Similarity: 0.4135335191718863

texts = ['公平','普惠']
# Similarity: 0.6672103544722907

texts = ['公平','盈利']
# Similarity: 0.49231586000301175
resp = dashscope.TextEmbedding.call(
    api_key= 'sk-687fa953ec64425a951a379b415a49c6',
    model=dashscope.TextEmbedding.Models.text_embedding_v4,
    input=texts,
    dimension=1024,
    output_type="dense"
)

if resp.status_code == HTTPStatus.OK:

    vec1 = np.array(
        resp.output["embeddings"][0]["embedding"]
    )

    vec2 = np.array(
        resp.output["embeddings"][1]["embedding"]
    )

    similarity = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1)
        * np.linalg.norm(vec2)
    )

    print("Similarity:", similarity)

else:
    print(resp)