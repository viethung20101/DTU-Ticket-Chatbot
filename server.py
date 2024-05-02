import socketio
import eventlet
from chain import *
from __init import *
from main import *

sio = socketio.Server(cors_allowed_origins='http://localhost:3000', ping_timeout=300, ping_interval=15)

clientSid = {}

def setupChat(sid):
    if sid in clientSid:
        print("@@@@Da co " + sid)
        return 
    else:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        clientSid[sid] = {
            'memory': memory,
            'stage_analyzer_inception_chain': setup_stage_analyzer_inception_chain(llm_checker,memory),
            'data_chain': setup_data_chain(llm_data,vectorstore,memory),
            'conversation_chain': setup_conversation_chain(llm_chat,memory)
        }
        return

def genAnswer(question, stage_analyzer_inception_chain, data_chain, conversation_chain):
    return invoke_chain(input=question,
                        stage_analyzer_inception_chain=stage_analyzer_inception_chain,
                        data_chain=data_chain,
                        conversation_chain=conversation_chain)

@sio.event
def connect(sid, environ):
    print('********************Client connected: ', sid)
    setupChat(sid) 

@sio.on('message')
def hanldeMessage(sid, data):
    print("@@@@Memory ", clientSid[sid]['memory'])
    answer = genAnswer(
        question=data['message'],
        stage_analyzer_inception_chain=clientSid[sid]['stage_analyzer_inception_chain'],
        data_chain=clientSid[sid]['data_chain'],
        conversation_chain=clientSid[sid]['conversation_chain']
    )
    sio.emit('message', {'message': answer}, room=sid)
    
@sio.event
def disconnect(sid):
    print('---------------------Client disconnected: ', sid)
    clientSid.pop(sid, None)

app = socketio.WSGIApp(sio)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('localhost', 6969)), app)
