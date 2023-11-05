import sys
from sdk.api.message import Message
from sdk.exceptions import CoolsmsException

class SmsSender:
    def __init__(self):
        self.api_key = "NCSIJ1FFM1LWFUOJ"
        self.api_secret = "S7TFTDQ1CBJF8V6NSMWXLDN3NK8SMC5Y"
        self.cool = Message(self.api_key, self.api_secret)
    
    def send_sms(self, to, from_, text):
        params = dict()
        params['type'] = 'sms' 
        params['to'] = to 
        params['from'] = from_
        params['text'] = text

        try:
            response = self.cool.send(params)
            print("Success Count : %s" % response['success_count'])
            print("Error Count : %s" % response['error_count'])
            print("Group ID : %s" % response['group_id'])

            if "error_list" in response:
                print("Error List : %s" % response['error_list'])

        except CoolsmsException as e:
            print("Error Code : %s" % e.code)
            print("Error Message : %s" % e.msg)
