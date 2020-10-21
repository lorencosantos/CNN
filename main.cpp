#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "cnn.h"
#include "sleep.h"

using namespace std;

float treinamento( vector<camada_t*>& camadas, tensor_t<float>& entrada, tensor_t<float>& target )
{
	// aplica a entrada na primeira camada e depois aplica a saida da camada anterior na proxima camada
	for ( int i = 0; i < camadas.size(); i++ )
	{
		if ( i == 0 )
			ativa( camadas[i], entrada );
		else
			ativa( camadas[i], camadas[i - 1]->saida );
	}

	// gradiente da ultima camada = saida - target
	tensor_t<float> grads = camadas.back()->saida - target;

	// backpropagation do gradiente
	for ( int i = camadas.size() - 1; i >= 0; i-- )
	{
		if ( i == camadas.size() - 1 )
			calc_grads( camadas[i], grads );
		else
			calc_grads( camadas[i], camadas[i + 1]->grads_entrada );
	}

	// corrige os pesos de todas as camadas
	for ( int i = 0; i < camadas.size(); i++ )
	{
		corrige_pesos( camadas[i] );
	}

	// calcula o erro
	float err = 0;
	for ( int i = 0; i < grads.tamanho.x * grads.tamanho.y * grads.tamanho.z; i++ )
	{
		float f = target.dados[i];
		if ( f > 0.5 )
			err += abs(grads.dados[i]);
	}
	return err * 100;
}


// faz o forward propagation
void forward( vector<camada_t*>& camadas, tensor_t<float>& entrada )
{
	// aplica a entrada na primeira camada, depois aplica a saida de cada camada na entrada da proxima
	for ( int i = 0; i < camadas.size(); i++ )
	{
		if ( i == 0 )
			ativa( camadas[i], entrada );
		else
			ativa( camadas[i], camadas[i - 1]->saida );
	}
}

// estrutura de cada caso de treinamento. cada caso possui a entrada e o target
struct case_t
{
	tensor_t<float> entrada;
	tensor_t<float> target;
};

// funcao para ler um arquivo
uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
	streamsize size = file.tellg();
	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	return buffer;
}

// funcao para ler o banco de dados de treinamento
vector<case_t> read_test_cases()
{
	// cria um vetor de cases
	vector<case_t> cases;

	// le as imagens de treinamento
	uint8_t* train_image = read_file( "train-images.idx3-ubyte" );

	// le os targets
	uint8_t* train_labels = read_file( "train-labels.idx1-ubyte" );

	// salva o numero de cases
	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	// para cada case, cria o tensor de dados de entrada e de target
	for ( int i = 0; i < case_count; i++ )
	{
		// cada imagem de entrada eh um tensor com 28 x 28 x 1 (matriz 2d)
		// cada target eh um tensor 10 x 1 x 1 (vetor 1d)
		case_t c {tensor_t<float>( 28, 28, 1 ), tensor_t<float>( 10, 1, 1 )};

		// pega a posicao inicial da imagem atual
		uint8_t* img = train_image + 16 + i * (28 * 28);

		// pega a posicao inicial do target atual
		uint8_t* label = train_labels + 8 + i;

		// coloca os pixels serializados no tensor c.entrada e normaliza de 0-255 para 0-1
		for ( int x = 0; x < 28; x++ )
			for ( int y = 0; y < 28; y++ )
				c.entrada( x, y, 0 ) = img[x + y * 28] / 255.f;

		// coloca o target em c.target na posicao correta do digito
		for ( int b = 0; b < 10; b++ )
			c.target( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

		cases.push_back( c );
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}

int main()
{
	vector<case_t> cases = read_test_cases();

	vector<camada_t*> camadas;

	// monta a arquitetura da rede: CONV -> RELU -> POOLING -> FC -> OUT

	// camada convolucional com passo = 1; filtros 5x5; 8 filtros; entrada de tamanho 28x28
	// sua saida sera 24 x 24 x 8 --> 8 imagens 24x24
	camada_conv_t * layer1 = new camada_conv_t( 1, 5, 8, cases[0].entrada.tamanho );		// 28 * 28 * 1 -> 24 * 24 * 8

	// camada relu com tamanho da entrada igual ao tamanho da saida da camada anterior
	camada_relu_t * layer2 = new camada_relu_t( layer1->saida.tamanho );

	// camada de pooling com passo 2, filtros 2x2 e entrada com tamanho igual ao tamanho da saida da camada anteior
	// entrada sera 24x24x8 e a saida sera 12x12x8
	camada_pool_t * layer3 = new camada_pool_t( 2, 2, layer2->saida.tamanho );				// 24 * 24 * 8 -> 12 * 12 * 8

	// camada fc com entrada de tamanho igual ao tamanho da saida da camada anterior e saida de tamanho 10
	// entrada sera 12 x 12 x 8 e saida sera 10
	camada_fc_t * layer4 = new camada_fc_t(layer3->saida.tamanho, 10);					// 12 * 12 * 8 -> 10

	camadas.push_back( (camada_t*)layer1 );
	camadas.push_back( (camada_t*)layer2 );
	camadas.push_back( (camada_t*)layer3 );
	camadas.push_back( (camada_t*)layer4 );


	// comeca o treinamento
	float amse = 0; // media temporal do erro quadratico medio
	int ic = 0; // medidor de milhar de decada

	cout << "Treinando a rede:" << endl;
	cout << "---------" << endl;
	// maximo 100000 epocas
	for ( int ep = 1; ep <= 2; ep++ )
	{
		cout << "Epoca "<< ep << " de 2" << endl;
		// para cada caso
		ic = 0;
		for ( case_t& t : cases )
		{
			// faz o treinamento e soma o erro
			float xerr = treinamento( camadas, t.entrada, t.target );
			amse += xerr;
			ic++;
			if ( ic % 10000 == 0 )
				cout << "\t--------Amostra " << ic << endl;
		}

		cout << "\t-------------------------- Erro = " << amse/(ep*ic) << endl;
	}


	// apos o treinamento, le a imagem de teste e faz sua classificacao
	// repete o processo a cada 1 segundo

	while ( true )
	{
		uint8_t * data = read_file( "test.ppm" );

		if ( data )
		{
			uint8_t * usable = data;

			while ( *(uint32_t*)usable != 0x0A353532 )
				usable++;

			// transforma a imagem RGB em monocromatica

#pragma pack(push, 1)
			struct RGB
			{
				uint8_t r, g, b;
			};
#pragma pack(pop)

			RGB * rgb = (RGB*)usable;

			tensor_t<float> image(28, 28, 1);
			for ( int i = 0; i < 28; i++ )
			{
				for ( int j = 0; j < 28; j++ )
				{
					RGB rgb_ij = rgb[i * 28 + j];
					image( j, i, 0 ) = (((float)rgb_ij.r
							     + rgb_ij.g
							     + rgb_ij.b)
							    / (3.0f*255.f));
				}
			}

			// faz o feed forward
			forward( camadas, image );
			tensor_t<float>& out = camadas.back()->saida;

			// verifica qual foi a saida de digitos entre 0 e 9
			for ( int i = 0; i < 10; i++ )
			{
				printf( "[%i] %f\n", i, out( i, 0, 0 )*100.0f );
			}

			delete[] data;
		}

		// espera 1 segundo para tesdar novamente
		// tempo para o usuario trocar a imagem de teste
		struct timespec wait;
		wait.tv_sec = 1;
		wait.tv_nsec = 0;
		nanosleep(&wait, nullptr);
	}
	return 0;
}
