#include "StorageInterface.h"

//Remote implementation
//contstructor
RemoteConnector::~RemoteConnector(){
	if(this->message != nullptr) free(this->message);
}

RemoteConnector::RemoteConnector(){
	this->portno = 80;
	this->host = "ashed.engr.uga.edu";
	this->message_size=0;

	this->sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd<0){
		perror("error creating socket");
	}

	this->server = gethostbyname(this->host.c_str());
	if(this->server == NULL){
		perror("error getting host IP");
	}

	memset(&this->serv_addr,0,sizeof(serv_addr));
	this->serv_addr.sin_family = AF_INET;
	this->serv_addr.sin_port = htons(portno);

	if(connect(this->sockfd,(struct sockaddr *)&this->serv_addr, sizeof(this->serv_addr))<0){
		perror("error connecting");
	}
}
//saves current Complex to remote DB
void RemoteConnector::writeParticles(){

	this->message_size=0;

	this->message_size+=strlen("POST  HTTP/1.0\r\n");
	this->message_size+=strlen("ashed.engr.uga.edu/createProtein?id=");
	this->message_size+=strlen(this->id.c_str());
	this->message_size+=strlen("Content-Type: application/x-www-form-urlencoded\r\n");
	this->message_size+=strlen("Content-Length: \r\n");
	std::string proteinJSON = createMolecularJSON(this->complex->molecules, this->complex->identifier);
	this->message_size+=strlen(std::to_string(proteinJSON.length()).c_str());
	this->message_size+=strlen("\r\n");
	this->message_size+=strlen(proteinJSON.c_str());

	this->message = (char*)malloc(this->message_size);

	sprintf(this->message, "POST ashed.engr.uga.edu/createProtein?id=%s HTTP/1.0\r\n",std::to_string(proteinJSON.length()).c_str());
	strcat(this->message, "Content-Type: application/x-www-form-urlencoded\r\n");
	sprintf(this->message+strlen(this->message), "Content-Length: %u\r\n",(unsigned int) proteinJSON.length());
	strcat(this->message,"\r\n");
	strcat(this->message, proteinJSON.c_str());

	this->total = strlen(this->message);
	this->sent = 0;
	do{
		this->bytes = write(this->sockfd,this->message+this->sent,this->total-this->sent);
		if(this->bytes<0){
			perror("error writing to socket");
		}
		if(this->bytes==0){
			break;
		}
		this->sent+=this->bytes;
	}while(this->sent<this->total);

	memset(this->response, 0, sizeof(this->response));
	this->total = sizeof(this->response)-1;
	this->received = 0;
	do{
		this->bytes = read(this->sockfd,this->response+this->received,this->total-this->received);
		if(this->bytes<0){
			perror("error reading response from socket");
		}
		if(this->bytes==0){
			break;
		}
		this->received+=this->bytes;
	}while(this->received<this->total);
}
//gets the triangle table from remote DB
std::vector<Triangle> RemoteConnector::readTriangles(){
	std::vector<Triangle> triangles;
	//READ THE TRIANGLES from DB
	return triangles;
}
//get paritcles table from remote DB
ParticleList* RemoteConnector::readParticles(){
	ParticleList* complex = new ParticleList();
	//the above should be replaced or just filled by DB complex
	return complex;
}
//saves/updates triangle table from remote DB
void RemoteConnector::writeTriangles(){

	this->message_size=0;

	this->message_size+=strlen("POST  HTTP/1.0\r\n");
	this->message_size+=strlen("ashed.engr.uga.edu/updateTriangle");
	this->message_size+=strlen("Content-Type: application/x-www-form-urlencoded\r\n");
	this->message_size+=strlen("Content-Length: \r\n");
	std::string trianglesString = createTrianglesJSON(this->triangles);
	this->message_size+=strlen(trianglesString.c_str());
	this->message_size+=strlen("\r\n");
	this->message_size+=strlen(trianglesString.c_str());

	this->message = (char*)malloc(this->message_size);

	sprintf(this->message, "POST ashed.engr.uga.edu/updateTriangle?id=%s HTTP/1.0\r\n",std::to_string(trianglesString.length()).c_str());
	strcat(this->message, "Content-Type: application/x-www-form-urlencoded\r\n");
	sprintf(this->message+strlen(this->message), "Content-Length: %u\r\n",(unsigned int) trianglesString.length());
	strcat(this->message,"\r\n");
	strcat(this->message, trianglesString.c_str());

	this->total = strlen(this->message);
	this->sent = 0;
	do{
		this->bytes = write(this->sockfd,this->message+this->sent,this->total - this->sent);
		if(this->bytes<0){
			perror("error writing to socket");
		}
		if(this->bytes==0){
			break;
		}
		this->sent+=this->bytes;
	}while(this->sent<this->total);

	memset(this->response, 0, sizeof(this->response));
	this->total = sizeof(this->response)-1;
	this->received = 0;
	do{
		this->bytes = read(this->sockfd,this->response+this->received,this->total - this->received);
		if(this->bytes<0){
			perror("error reading response from socket");
		}
		if(this->bytes==0){
			break;
		}
		this->received+=this->bytes;
	}while(this->received<this->total);

}

//Local db implementation goes here
LocalConnector::LocalConnector(){

}
LocalConnector::~LocalConnector(){
}

//saves current Complex to local DB
void LocalConnector::writeParticles(){

}
//gets the triangle table from local DB
std::vector<Triangle> LocalConnector::readTriangles(){
	std::vector<Triangle> triangles;
	//READ THE TRIANGLES from DB
	return triangles;
}
//get paritcles table from local DB
ParticleList* LocalConnector::readParticles(){
	ParticleList* complex = new ParticleList();
	//the above should be replaced or just filled by DB complex
	return complex;
}

//saves/updates triangle table from local DB
void LocalConnector::writeTriangles(){


}

//local file io implementation
//contstructor
LocalFileConnector::LocalFileConnector(){

}
LocalFileConnector::~LocalFileConnector(){
}


//saves current Complex to local json
void LocalFileConnector::writeParticles(){
	std::string proteinJSON = createMolecularJSON(this->complex->molecules, this->complex->identifier);
	std::ofstream jsonOutput("data/complexJsons/"+ this->id + ".json");
	if(jsonOutput.is_open()){
		jsonOutput << proteinJSON; //write the json
		jsonOutput.close();
	}
	else{
		std::cout<<"ERROR creating "<<this->id<<".json"<<std::endl;
		exit(-1);
	}

	std::cout << this->id + ".json" << " has been created.\n---------------------------------------------------" << std::endl;
}

//gets the triangles from local json
std::vector<Triangle> LocalFileConnector::readTriangles(){
	this->triangles.clear();
	std::string filePath = "./data/triangles.json";
	if(!fileExists(filePath)) return this->triangles;
	std::string triangleStr =  getStringFromJSON(filePath);
	readTrianglesJSON(triangleStr, this->triangles);
	return this->triangles;
}

//get particles from local json
ParticleList* LocalFileConnector::readParticles(){
	std::string filePath = "./data/complexJsons/"+this->id+".json";
	std::string particlesStr = getStringFromJSON(filePath);
	this->complex = new ParticleList();
	readMolecularJSON(particlesStr, this->complex->molecules, this->complex->identifier);
	return this->complex;
}

void LocalFileConnector::writeTriangles(){

	std::string trianglesStr = createTrianglesJSON(this->triangles);

	std::ofstream jsonOutput("data/triangles.json");
	if(jsonOutput.is_open()){
		std::cout << "Printing JSON" << std::endl;
		jsonOutput << trianglesStr;
		jsonOutput.close();
	}
	else{
		std::cout<<"ERROR creating triangles.json"<<std::endl;
		exit(-1);
	}

	std::cout <<"triangles.json" << " has been created.\n---------------------------------------------------" << std::endl;
}
