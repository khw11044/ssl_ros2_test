# Sound Source Localization Result Message
# 각도 범위: -90 ~ +90 (가운데 0, 음수=왼쪽, 양수=오른쪽)

std_msgs/Header header

# GCC 결과
int32 gcc_angle              # 현재 각도 (-90 ~ +90)
string gcc_direction         # "Left", "Front", "Right"
float32 gcc_db               # 현재 dB
bool gcc_event               # 이벤트 발생 여부
int32 gcc_event_angle        # 이벤트 각도 (event=true일 때만 유효)
string gcc_event_direction   # 이벤트 방향 "Left", "Front", "Right"
float32 gcc_event_db         # 이벤트 시 dB

# SRP 결과
float32[] srp_distribution   # -90~+90 분포 배열
int32 srp_peak_percent       # 최대값 퍼센트 (0~100)
int32 srp_peak_angle         # 최대값 각도 (-90 ~ +90)
string srp_direction         # "Left", "Front", "Right"
