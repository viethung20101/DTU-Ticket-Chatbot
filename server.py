import socketio
import eventlet

sio = socketio.Server()

@sio.event
def connect(sid, environ):
    print('Client connected', sid)
    
@sio.on('message')
def hanlde_message(sid, data):
    print('okio')
    question = data['message']
    print(question)
    sio.emit('message', {'message': question})

@sio.event
def disconnect(sid):
    print('Client disconnected', sid)

app = socketio.WSGIApp(sio)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('localhost', 6969)), app)
