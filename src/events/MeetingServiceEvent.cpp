#include "MeetingServiceEvent.h"
#include "../util/WebSocketClient.h"
#include "../Zoom.h"
#include "../util/json.hpp"

using json = nlohmann::json;
extern WebSocketClient* g_webSocketClient;

void cleanup() {
    // Perform any necessary cleanup tasks here
    if (g_webSocketClient) {
        g_webSocketClient->close(); // Close the WebSocket connection
        delete g_webSocketClient;   // Delete the WebSocket client object
        g_webSocketClient = nullptr;
    }
}


void MeetingServiceEvent::onMeetingStatusChanged(MeetingStatus status, int iResult) {
    if (m_onMeetingStatusChanged) {
        m_onMeetingStatusChanged(status, iResult);
        return;
    }

    stringstream ss;
    ss << iResult;
    auto result = ss.str();

    std::string meetingID; // Declare meetingID outside the switch statement
    auto* zoom = &Zoom::getInstance();
    switch (status) {
        case MEETING_STATUS_CONNECTING:
            Log::info("connecting to the meeting");
            break;
        case MEETING_STATUS_RECONNECTING:
            Log::info("reconnecting to the meeting");
            break;
        case MEETING_STATUS_DISCONNECTING:
            Log::info("disconnecting from the meeting");
            break;
        case MEETING_STATUS_INMEETING:
            Log::success("connected");
            if (m_onMeetingJoin) m_onMeetingJoin();
            return;
        case MEETING_STATUS_ENDED:
            Log::success("meeting ended");
            if (m_onMeetingEnd) m_onMeetingEnd();
            meetingID = Zoom::getInstance().getMeetingID();
            // Send the meeting ID to the WebSocket
            if (g_webSocketClient) {
                nlohmann::json message = {
                    {"action", "meeting_ended"},
                    {"meeting_id", meetingID}
                };
                g_webSocketClient->send(message.dump());
            }
            cleanup(); // Perform cleanup before exiting
            zoom->leave();
            zoom->clean();
            exit(0);
            return;
        case MEETING_STATUS_FAILED:
            Log::error("failed to connect to the meeting with MeetingFailCode " + result);
            break;
        case MEETING_STATUS_WAITINGFORHOST:
            Log::info("waiting for the meeting to start");
            break;
        default:
            Log::error("unknown meeting status");
            break;
    }
}

void MeetingServiceEvent::setOnMeetingJoin(const function<void()>& callback) {
    m_onMeetingJoin = callback;
}

void MeetingServiceEvent::setOnMeetingEnd(const function<void()>& callback) {
    m_onMeetingEnd = callback;
}

void MeetingServiceEvent::setOnMeetingStatusChanged(const function<void(MeetingStatus, int)>& callback) {
    m_onMeetingStatusChanged = callback;
}
